#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/STLExtras.h"

#include "triton/Conversion/TritonGPUToLLVM/CuteJitOrc.h"

// 保存并屏蔽 Triton 宏，避免污染 CUTE/CUTLASS
#pragma push_macro("select")
#pragma push_macro("bitcast")

#ifdef select
  #undef select
  #define TRITON_RESTORE_SELECT 1
#endif

#ifdef bitcast
  #undef bitcast
  #define TRITON_RESTORE_BITCAST 1
#endif

// 这里随便 include 你需要的 CUTE/CUTLASS 头
#include "cute/layout.hpp"
// 如果还需要别的：#include <cute/print.hpp> 等

// 恢复宏定义
#ifdef TRITON_RESTORE_SELECT
  #pragma pop_macro("select")
  #undef TRITON_RESTORE_SELECT
#endif

#ifdef TRITON_RESTORE_BITCAST
  #pragma pop_macro("bitcast")
  #undef TRITON_RESTORE_BITCAST
#endif

using namespace cute;

namespace mlir {

namespace triton::gpu {
Type getFunctionType(Type resultType, ValueRange operands) {
  SmallVector<Type> operandTypes(operands.getTypes());
  return LLVM::LLVMFunctionType::get(resultType, operandTypes);
}

LLVM::LLVMFuncOp appendOrGetExternFuncOp(RewriterBase &rewriter, Operation *op,
                                         StringRef funcName, Type funcType,
                                         StringRef libname /*= ""*/,
                                         StringRef libpath /*= ""*/) {
  using LLVM::LLVMFuncOp;

  auto funcAttr = StringAttr::get(op->getContext(), funcName);
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
  if (funcOp)
    return cast<LLVMFuncOp>(*funcOp);

  Operation *parent = op;
  if (!isa<LLVM::LLVMFuncOp>(op))
    parent = op->getParentOfType<LLVM::LLVMFuncOp>();
  OpBuilder b(parent);
  auto ret = b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  ret.getOperation()->setAttr("libname",
                              StringAttr::get(op->getContext(), libname));
  ret.getOperation()->setAttr("libpath",
                              StringAttr::get(op->getContext(), libpath));
  return ret;
}
} // namespace triton::gpu

SmallVector<std::pair<StringAttr, Value>>
applyLinearLayout(Location loc, RewriterBase &rewriter,
                  const LinearLayout &layout,
                  ArrayRef<std::pair<StringAttr, Value>> indices) {
  assert(layout.getNumInDims() == indices.size());
  for (auto [inDimName, idx] : indices) {
    assert(layout.hasInDim(inDimName) && "Invalid inDimName");
  }

  // This function can emit a lot of MLIR code, which ultimately makes
  // compilation slow.  (We think this shouldn't be the case -- it's not *that*
  // much code -- but we're not clear on how to fix the slowness, which happens
  // in the bowels of MLIR.)
  //
  // As a result we go through some contortions to avoid emitting code where
  // possible.

  // Manually constant-fold the layout where possible.
  SmallVector<std::pair<StringAttr, int32_t>> constantIns;
  for (auto [inDimName, idx] : indices) {
    if (auto constant = idx.getDefiningOp<LLVM::ConstantOp>()) {
      constantIns.push_back(
          {inDimName, cast<IntegerAttr>(constant.getValue()).getInt()});
    } else {
      constantIns.push_back({inDimName, 0});
    }
  }
  SmallVector<int32_t> constantComponent =
      llvm::to_vector(llvm::make_second_range(layout.apply(constantIns)));

  Value zero = i32_val(0);
  SmallVector<std::pair<StringAttr, Value>> outIndices;
  for (auto [i, outDimName] : llvm::enumerate(layout.getOutDimNames())) {
    if (constantComponent[i] == 0)
      outIndices.push_back({outDimName, zero});
    else
      outIndices.push_back({outDimName, i32_val(constantComponent[i])});
  }

  for (auto [inDimName, idx] : indices) {
    if (idx.getDefiningOp<LLVM::ConstantOp>()) {
      continue;
    }

    int nBits = layout.getInDimSizeLog2(inDimName);
    for (int i = 0; i < nBits; i++) {
      Value bit = and_(idx, i32_val(1 << i));
      Value bit_is_zero = icmp_eq(bit, zero);
      for (auto &[outDimName, outIdx] : outIndices) {
        int32_t basis = layout.getBasis(inDimName, i, outDimName);
        if (basis == 0)
          continue;
        outIdx = xor_(outIdx, select(bit_is_zero, zero, i32_val(basis)));
      }
    }
  }

  return outIndices;
}

// 只改模板参数即可复用：N=Tensor元素数，S=元素stride（通常1）
// W=warp数，T=每warp线程数（32），V=每线程连续元素数
template<int N, int S, int W, int T, int V>
void dump_offsets(int warpId, int laneId) {
  llvm::outs() << "[ZSY-CUTE-LAYOUT]: ";
  static_assert((N % (W*T*V)) == 0, "N must be divisible by W*T*V");
  constexpr int R = N / (W*T*V);  // “翻页”个数

  // 1) GMEM 一维布局：索引 -> 物理地址（stride = S）
  constexpr auto GMEM =
      make_layout(make_shape(Int<N>{}), make_stride(Int<S>{}));

  // 2) 线程×向量 元素：shape=(T,V), stride=(V,1)
  constexpr auto L_lane_val =
      make_layout(make_shape(Int<T>{}, Int<V>{}),
                  make_stride(Int<V>{},  Int<1>{}));

  // 3) 加 warp：shape=(W,T,V), stride=(T*V, V, 1)
  constexpr auto L_warp_lane_val =
      make_layout(make_shape(Int<W>{}, Int<T>{}, Int<V>{}),
                  make_stride(Int<(T*V)>{}, Int<V>{}, Int<1>{}));

  // 4) 再加翻页 reg_hi：shape=(W,T,R,V),
  //    stride=(T*V, V, W*T*V, 1)   // 这里没有写死128/512，均由参数计算
  constexpr auto L_full =
      make_layout(make_shape(Int<W>{}, Int<T>{}, Int<R>{}, Int<V>{}),
                  make_stride(Int<(T*V)>{}, Int<V>{}, Int<(W*T*V)>{}, Int<1>{}));

  // 5) 组合得到： (warp,lane,reg_hi,val) -> 物理地址
  constexpr auto MAP = composition(GMEM, L_full);

  // 打印 (warpId,laneId) 的所有寄存器元素偏移（共 R*V 个）
  for (int reg = 0; reg < R*V; ++reg) {
    int reg_hi = reg / V;      // 第几“页”
    int val    = reg % V;      // 页内第几个连续元素
    int off = int(MAP(warpId, laneId, reg_hi, val));
    std::cout << off << (reg+1 < R*V ? ' ' : '\n');
  }
}

SmallVector<SmallVector<Value>>
emitIndices(Location loc, RewriterBase &rewriter, const TargetInfoBase &target,
            Attribute layout, RankedTensorType type, bool withCTAOffset) {
  MLIRContext *ctx = rewriter.getContext();
  auto shape = type.getShape();
  llvm::outs() << "[ZSY-LinearLayout] blocked_tensor_layout = " << layout << "\n";

  std::optional<LinearLayout> ll = triton::gpu::toLinearLayout(shape, layout);
  if (!ll.has_value())
    llvm::report_fatal_error("Failed to convert layout to linear layout");

  // TODO(jlebar): We could add strong typing if we wanted; for now this is
  // "stringly typed".
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");
  StringAttr kBlock = str_attr("block");

  Value threadId = getThreadId(rewriter, loc);
  Value threadsPerWarp = i32_val(ll->getInDimSize(kLane));
  Value laneId = urem(threadId, threadsPerWarp);
  Value warpId = udiv(threadId, threadsPerWarp);
  Value blockId =
      withCTAOffset ? target.getClusterCTAId(rewriter, loc) : i32_val(0);
  unsigned rank = shape.size();
  SmallVector<SmallVector<Value>> ret;
  // Linear layout function is split in two parts below:
  // L(r, t, w, b) = L(0, t, w, b) xor L(r, 0, 0, 0)
  //     idxs      =    idxsBase   xor    idxsReg
  //
  // L(0, t, w, b) part is the same for all registers,
  // so we hoist it out of the main register loop in the below.
  //
  // This approach produces code with lower register pressure and
  // less computations, compared to fused L(r,t,w,b) method.
  // auto idxsBase = applyLinearLayout(loc, rewriter, *ll,
  //                                   {{kRegister, i32_val(0)},
  //                                    {kLane, laneId},
  //                                    {kWarp, warpId},
  //                                    {kBlock, blockId}});

  SmallVector<std::pair<StringAttr, Value>> idxsBase;
  Value idx_tmp0 = getThreadId(rewriter, loc);
  Operation *anchor = rewriter.getInsertionBlock()->getParentOp();

  // Value idx_tmp1 = shl(idx_tmp0, i32_val(2));
  LLVM::LLVMFuncOp nvCuteFuncOp =
    triton::gpu::appendOrGetExternFuncOp(rewriter, anchor, "__nv_cute_get_idx_base",
      triton::gpu::getFunctionType(idx_tmp0.getType(), {idx_tmp0}));
  Value idx_tmp1 = LLVM::createLLVMCallOp(rewriter, loc, nvCuteFuncOp, idx_tmp0).getResult();
  // I implement only the 1D case
  idxsBase.push_back({str_attr("dim0"), idx_tmp1});

  auto blockedLayout = dyn_cast<triton::gpu::BlockedEncodingAttr>(layout);
  const auto &order = blockedLayout.getOrder();
  // I implement only the 1D case
  assert(order.size() == 1);
  assert(shape.size() == 1);
                                     auto sizePerThread = blockedLayout.getSizePerThread()[order[0]];
  llvm::outs() << "sizePerThread = " << sizePerThread << "\n";
  auto THREADSPERWARP = blockedLayout.getThreadsPerWarp()[order[0]];
  llvm::outs() << "THREADSPERWARP = " << THREADSPERWARP << "\n";
  auto warpsPerCTA = blockedLayout.getWarpsPerCTA()[order[0]];
  llvm::outs() << "warpsPerCTA = " << warpsPerCTA << "\n";
  auto tensorShapeSize = shape[order[0]];
  llvm::outs() << "tensorShapeSize = " << tensorShapeSize << "\n";

  auto st = triton::runtime::initCuteJitORC(sizePerThread, THREADSPERWARP, warpsPerCTA, tensorShapeSize);
  if (!st.ok) {
    llvm::outs() << "[CuteJIT/MCJIT] init failed: " << st.msg << "\n";
  }
  int cuteLayoutSize = triton::runtime::getCuteLayoutSize();
  if (-1 == cuteLayoutSize)
    llvm::outs() << "[CuteJIT/MCJIT] getLayoutSize failed\n";
  std::vector<int> resultBuffer;
  resultBuffer.reserve(cuteLayoutSize);
  st = triton::runtime::calculateCuteLayout(resultBuffer.data());
  if (!st.ok) {
    llvm::outs() << "[CuteJIT/MCJIT] calculate failed: " << st.msg << "\n";
  }
  llvm::outs() << "[Triton-CuteJIT/MCJIT] index offset = ";
  for (int i = 0; i < cuteLayoutSize; i++) {
    llvm::outs() << resultBuffer[i] << " ";
  }
  llvm::outs() << "\n";

#if 0
  auto thr_layout = make_layout(make_shape(Int<4 * 32>{}), make_stride(Int<1>{}));
  auto val_layout = make_layout(make_shape(Int<4>{}), make_stride(Int<1>{}));
  auto layout_mn = raked_product(thr_layout, val_layout);       // (tid,val) ⟶ (M,N)
  printf("layout_mn: ");     print(layout_mn);  printf("\n");
  auto tiler_mn  = product_each(cute::shape(layout_mn));              // (TileM, TileN) 或 1D 时为 (Tile)
  // 临时 (tid,val) 的列主序布局：stride=(1, thr_size)
  auto tmp_tv = make_layout(
      make_shape(size(thr_layout), size(val_layout)),
      make_stride(Int<1>{}, size(thr_layout)));
  printf("tmp_tv: "); print(tmp_tv); printf("\n");
  // TV-layout: (tid,val) -> (M,N)
  auto tv_layout = composition(tmp_tv, right_inverse(layout_mn));
  printf("inv(layout_mn): "); print(right_inverse(layout_mn)); printf("\n");

  // 打印：和 cutedsl 的 `print` 对齐
  printf("Tiler: ");     print(tiler_mn);  printf("\n");
  printf("TV Layout: "); print(tv_layout); printf("\n");

  auto GMEM_1D = make_layout(make_shape(Int<2048>{}), make_stride(Int<1>{}));
  auto tiles = zipped_divide(GMEM_1D, tiler_mn);

  // auto LL = composition(layout2, layout1);
  // auto LL = layout2.compose(layout1, _);
  auto L = logical_divide(tiles, tv_layout);
  // auto LL = composition(tiles, make_tile(tv_layout, _));
  printf("L: "); print(L); printf("\n");

  // 演示：tid=0，打印 val=0..3, page=0..1 的 8 个偏移
  int tid = 1;
  for (int p = 0; p < int(size(get<1>(L))); ++p) {
    for (int v = 0; v < int(size(get<1>(get<0>(L)))); ++v) {
      int off = L(make_coord( make_coord(tid, v), p ));     // C++ 里 layout 是可调用的
      printf("%d ", off);
    }
  }
  printf("\n");

#endif

  // llvm::outs() << "[ZSY-LinearLayout] reg = " << ll->getInDimSize(str_attr("register")) << ":\n";
  // for (unsigned reg = 0; reg < ll->getInDimSize(str_attr("register")); reg++) {
  //   auto idxsReg =
  //       ll->apply({{kRegister, reg}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
  //   SmallVector<std::pair<StringAttr, Value>> idxs;
  //   for (auto [idxBase, idxReg] : llvm::zip(idxsBase, idxsReg)) {
  //     auto dimName = idxBase.first;
  //     assert(dimName == idxReg.first &&
  //            "dim names of block+warp+thread and register idx should be equal");
  //     auto idx = xor_(idxBase.second, i32_val(idxReg.second));
  //     idxs.emplace_back(dimName, idx);
  //     llvm::outs() << "idx(regId = " << reg << ") = " << idxReg.second << "\n";

  //   }

  //   assert(idxs.size() == rank);
  //   for (unsigned k = 0; k < rank; ++k) {
  //     assert(idxs[k].first == str_attr("dim" + std::to_string(k)));
  //   }
  //   ret.push_back(llvm::to_vector(llvm::make_second_range(idxs)));
  // }

  // Suppose I only implement 1D BlockedLayout
  for (int i = 0; i < cuteLayoutSize; i++) {
    ret.push_back(SmallVector<Value>{xor_(idx_tmp1 /* idxsBase */, i32_val(resultBuffer[i]))});
  }

  return ret;
}

bool emitTransferBetweenRegistersAndShared(
    RankedTensorType registerTy, MemDescType sharedTy, Type elemLlvmTy,
    std::optional<int32_t> maxVecElems, Value shmemBase,
    ArrayRef<Value> shmemStrides, Location loc, RewriterBase &rewriter,
    const TargetInfoBase &target,
    std::function<void(VectorType, Value /*shmemAddr*/)> perVectorCallback) {
  MLIRContext *ctx = rewriter.getContext();

  auto shape = registerTy.getShape();
  int rank = shape.size();

  StringAttr kBlock = str_attr("block");
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");

  std::optional<LinearLayout> regLayout =
      triton::gpu::toLinearLayout(shape, registerTy.getEncoding());
  std::optional<LinearLayout> sharedLayout = triton::gpu::toLinearLayout(
      shape, sharedTy.getEncoding(), elemLlvmTy.getIntOrFloatBitWidth());
  if (!regLayout.has_value() || !sharedLayout.has_value()) {
    return false;
  }
  auto sharedOrder = triton::gpu::getOrder(sharedTy.getEncoding());

  // sharedLayout's in-dims are currently (offset, block).  Reshape to
  // (offsetX1, offsetX2, ..., block) so that we can apply the N-dimensional
  // shmem strides.  (The offsetX's appear in minor-to-major order.)
  auto sharedLegacy =
      cast<triton::gpu::SharedEncodingAttr>(sharedTy.getEncoding());
  SmallVector<std::pair<StringAttr, int32_t>> multiDimSharedSize;
  for (int i = 0; i < rank; i++) {
    int dim = sharedOrder[i];
    int64_t size = std::max(
        int64_t{1},
        shape[dim] / sharedLegacy.getCTALayout().getCTASplitNum()[dim]);
    multiDimSharedSize.push_back(
        {str_attr("offset" + std::to_string(dim)), size});
  }
  multiDimSharedSize.push_back({kBlock, sharedLayout->getInDimSize(kBlock)});
  sharedLayout = sharedLayout->reshapeIns(multiDimSharedSize);

  // regToSharedLayout maps from (register, lane, warp, block) to (offsetX1,
  // ..., offsetXN, block), where the offsetX's are in minor-to-major order.
  LinearLayout regToSharedLayout = regLayout->invertAndCompose(*sharedLayout);

  // TODO(jlebar): We don't currently support loading from shared memory in a
  // different CTA.  We'd need to emit `mapa.shared::cluster` instructions.
  for (int inBlock = 1; inBlock < regToSharedLayout.getInDimSize(kBlock);
       inBlock *= 2) {
    auto idx = llvm::to_vector(llvm::make_second_range(regToSharedLayout.apply(
        {{kRegister, 0}, {kLane, 0}, {kWarp, 0}, {kBlock, inBlock}})));
    // offsetX1, ..., offsetXN must all be 0.
    if (!llvm::all_of(ArrayRef(idx).drop_back(1),
                      [&](auto offset) { return offset == 0; })) {
      return false;
    }
    // Check if there's any cross CTA load.
    int32_t outBlock = idx.back();
    if (outBlock != inBlock) {
      return false;
    }
  }

  // Determine how many consecutive registers map to consecutive shmem elements
  // in out-dimension offsetN.  This is our load instruction's vector width.
  //
  // It's OK if the vector width we choose here is wider than the hardware
  // supports; LLVM will legalize it.
  //
  // TODO(jlebar): shmemStrides are Values, but most of them are usually integer
  // constants.  We could add those constant strides to the LL, and then before
  // calling getNumConsecutiveInOut(), we could flatten consecutive out-dims
  // which have known strides.  This would allow us to vectorize across multiple
  // shmem out dimensions where possible.
  const int vecElems =
      std::min(regToSharedLayout.getNumConsecutiveInOut(),
               maxVecElems.value_or(std::numeric_limits<int>::max()));

  Value threadId = getThreadId(rewriter, loc);
  Value threadsPerWarp = i32_val(regToSharedLayout.getInDimSize(kLane));
  Value laneId = urem(threadId, threadsPerWarp);
  Value warpId = udiv(threadId, threadsPerWarp);

  int numElems = regToSharedLayout.getInDimSize(kRegister);
  auto vecTy = vec_ty(elemLlvmTy, vecElems);
  auto ptrTy = shmemBase.getType();
  Value zero = i32_val(0);
  SmallVector<Value> ret;
  for (int i = 0; i < numElems / vecElems; i++) {
    // Get the address to load/store.  The multi-dim address is (offsetX1, ...,
    // offsetXN, block), where the offsets appear in minor-to-major order, and
    // we drop_end to drop block, which we know from above will be 0.
    auto multiDimShmemOffset =
        llvm::to_vector(llvm::drop_end(llvm::make_second_range(
            applyLinearLayout(loc, rewriter, regToSharedLayout,
                              {{kRegister, i32_val(i * vecElems)},
                               {kLane, laneId},
                               {kWarp, warpId},
                               {kBlock, zero}}))));

    // Reorder strides according to `order`.  This way they match the
    // multi-dimensional offsets in regToSharedLayout.
    Value shmemOffset = dot(rewriter, loc, multiDimShmemOffset,
                            applyPermutation(shmemStrides, sharedOrder));
    auto vecAddr = gep(ptrTy, elemLlvmTy, shmemBase, shmemOffset);
    vecAddr.setInbounds(true);

    perVectorCallback(vecTy, vecAddr);
  }
  return true;
}

SmallVector<Value> loadSharedToDistributed(RankedTensorType dstTy,
                                           MemDescType srcTy, Type elemLlvmTy,
                                           SharedMemoryObject smemObj,
                                           Location loc, RewriterBase &rewriter,
                                           const TargetInfoBase &target) {
  SmallVector<Value> ret;
  bool success = emitTransferBetweenRegistersAndShared(
      dstTy, srcTy, elemLlvmTy, /*maxVecElems=*/std::nullopt, smemObj.getBase(),
      smemObj.getStrides(), loc, rewriter, target,
      [&](VectorType vecTy, Value vecAddr) {
        auto vecVal = load(vecTy, vecAddr);
        vecVal.setAlignment(vecTy.getNumElements() *
                            elemLlvmTy.getIntOrFloatBitWidth() / 8);

        for (int v = 0; v < vecTy.getNumElements(); v++) {
          ret.push_back(extract_element(elemLlvmTy, vecVal, i32_val(v)));
        }
      });
  if (!success)
    llvm::report_fatal_error("Failed to emit transfer from shared to register");

  return ret;
}

void storeDistributedToShared(MemDescType dstTy, RankedTensorType srcTy,
                              Type elemLlvmTy, ArrayRef<Value> srcVals,
                              Value smemBase, ArrayRef<Value> dstStrides,
                              Location loc, RewriterBase &rewriter,
                              const TargetInfoBase &target,
                              std::pair<size_t, Type> *const llvmOpCount) {
  bool success = emitTransferBetweenRegistersAndShared(
      srcTy, dstTy, elemLlvmTy, /*maxVecElems=*/std::nullopt, smemBase,
      dstStrides, loc, rewriter, target, [&](VectorType vecTy, Value vecAddr) {
        ArrayRef<Value> vals = srcVals.take_front(vecTy.getNumElements());
        srcVals = srcVals.drop_front(vecTy.getNumElements());

        Value vec = undef(vecTy);
        for (int i = 0; i < vals.size(); i++) {
          vec = insert_element(vec, vals[i], i32_val(i));
        }
        store(vec, vecAddr)
            .setAlignment(vecTy.getNumElements() *
                          elemLlvmTy.getIntOrFloatBitWidth() / 8);
        if (llvmOpCount) {
          ++(llvmOpCount->first);
          llvmOpCount->second = vecTy;
        }
      });

  if (!success)
    llvm::report_fatal_error("Failed to emit transfer from register to shared");
}

SmallVector<SmallVector<unsigned>> emitOffsetForLayout(Attribute layout,
                                                       RankedTensorType type) {
  MLIRContext *ctx = layout.getContext();
  auto shape = type.getShape();
  unsigned rank = shape.size();

  auto ll = triton::gpu::toLinearLayout(shape, layout);
  if (!ll.has_value())
    llvm::report_fatal_error("Unsupported layout");

  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");
  StringAttr kBlock = str_attr("block");

  SmallVector<SmallVector<unsigned>> offsets;
  for (int i = 0; i < ll->getInDimSize(str_attr("register")); i++) {
    auto idxs =
        ll->apply({{kRegister, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    assert(idxs.size() == rank);
    for (unsigned k = 0; k < rank; ++k) {
      assert(idxs[k].first == str_attr("dim" + std::to_string(k)));
    }
    offsets.push_back(
        llvm::to_vector_of<unsigned>(llvm::make_second_range(idxs)));
  }
  return offsets;
}

namespace LLVM {
using namespace mlir::triton;
using mlir::triton::gpu::getOrder;
using mlir::triton::gpu::getSizePerThread;

Value createConstantI1(Location loc, OpBuilder &rewriter, bool v) {
  auto i1ty = rewriter.getIntegerType(1);
  return rewriter.create<LLVM::ConstantOp>(loc, i1ty,
                                           IntegerAttr::get(i1ty, v));
}

Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v) {
  auto i32ty = rewriter.getIntegerType(32);
  return rewriter.create<LLVM::ConstantOp>(loc, i32ty,
                                           IntegerAttr::get(i32ty, v));
}

Value createConstantI64(Location loc, OpBuilder &rewriter, int64_t v) {
  auto i64ty = rewriter.getIntegerType(64);
  return rewriter.create<LLVM::ConstantOp>(loc, i64ty,
                                           IntegerAttr::get(i64ty, v));
}

Value createConstantF16(Location loc, OpBuilder &rewriter, float v) {
  auto type = type::f16Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getF16FloatAttr(v));
}

Value createConstantF32(Location loc, OpBuilder &rewriter, float v) {
  auto type = type::f32Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getF32FloatAttr(v));
}

Value createConstantF64(Location loc, OpBuilder &rewriter, double v) {
  auto type = type::f64Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getF64FloatAttr(v));
}

Value createNaNConstant(Location loc, OpBuilder &rewriter, Type type) {
  if (!isa<FloatType>(type)) {
    llvm::report_fatal_error("Creating NaN constant for non-float type!");
  }
  return rewriter.create<LLVM::ConstantOp>(
      loc, type, APFloat::getNaN(cast<FloatType>(type).getFloatSemantics()));
}

// Create an index type constant.
Value createIndexConstant(OpBuilder &builder, Location loc,
                          const TypeConverter *converter, int64_t value) {
  Type ty = converter->convertType(builder.getIndexType());
  return builder.create<LLVM::ConstantOp>(loc, ty,
                                          builder.getIntegerAttr(ty, value));
}

// Create an integer constant of \param width bits.
Value createLLVMIntegerConstant(OpBuilder &builder, Location loc, short width,
                                int64_t value) {
  Type ty = builder.getIntegerType(width);
  return builder.create<LLVM::ConstantOp>(loc, ty,
                                          builder.getIntegerAttr(ty, value));
}

LLVM::CallOp createLLVMCallOp(OpBuilder &builder, Location loc,
                              LLVMFuncOp funcOp, ValueRange args) {
  auto op = builder.create<LLVM::CallOp>(loc, funcOp, args);
  op.getProperties().setOpBundleSizes(builder.getDenseI32ArrayAttr({}));
  op.getProperties().setOperandSegmentSizes({static_cast<int>(args.size()), 0});
  return op;
}

LLVM::CallIntrinsicOp
createLLVMIntrinsicCallOp(OpBuilder &builder, Location loc, StringRef intrinsic,
                          TypeRange types, ValueRange args) {
  auto op = builder.create<LLVM::CallIntrinsicOp>(loc, types, args);
  op.getProperties().setIntrin(builder.getStringAttr(intrinsic));
  op.getProperties().setOpBundleSizes(builder.getDenseI32ArrayAttr({}));
  op.getProperties().setOperandSegmentSizes({static_cast<int>(args.size()), 0});
  return op;
}

bool isConstantZero(Value v) {
  if (auto constantOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
      return attr.getValue().isZero();
    }
    if (auto attr = dyn_cast<FloatAttr>(constantOp.getValue())) {
      return attr.getValue().isZero();
    }
  }
  return false;
}

SharedMemoryObject getSharedMemoryObjectFromStruct(Location loc,
                                                   Value llvmStruct,
                                                   Type elemTy,
                                                   RewriterBase &rewriter) {
  ArrayRef<Type> types =
      cast<LLVM::LLVMStructType>(llvmStruct.getType()).getBody();
  SmallVector<Value> elems(types.size());
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    elems[i] = extract_val(type, llvmStruct, i);
  }

  auto rank = (elems.size() - 1) / 2;
  return {/*base=*/elems[0],
          /*baseElemType=*/elemTy,
          /*strides=*/{elems.begin() + 1, elems.begin() + 1 + rank},
          /*offsets=*/{elems.begin() + 1 + rank, elems.end()}};
}

SmallVector<Value> getStridesFromShapeAndOrder(ArrayRef<int64_t> shape,
                                               ArrayRef<unsigned> order,
                                               Location loc,
                                               RewriterBase &rewriter) {
  auto rank = shape.size();
  SmallVector<Value> strides(rank);
  int64_t stride = 1;
  for (auto idx : order) {
    strides[idx] = i32_val(stride);
    stride *= shape[idx];
  }
  return strides;
}

// Convert an \param index to a multi-dim coordinate given \param shape and
// \param order.
SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape,
                               ArrayRef<unsigned> order) {
  unsigned rank = shape.size();
  assert(rank == order.size());
  auto reordered = applyPermutation(shape, order);
  SmallVector<Value> reorderedMultiDim(rank);
  if (auto constantOp = linear.getDefiningOp<arith::ConstantOp>()) {
    unsigned intVal = mlir::cast<IntegerAttr>(constantOp.getValue())
                          .getValue()
                          .getSExtValue();
    reorderedMultiDim = delinearize(rewriter, loc, intVal, reordered);
  } else {
    reorderedMultiDim = delinearize(rewriter, loc, linear, reordered);
  }
  SmallVector<Value> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               unsigned linear, ArrayRef<unsigned> shape) {
  unsigned rank = shape.size();
  assert(rank > 0);
  SmallVector<Value> multiDim(rank);
  unsigned remained = linear;
  for (auto &&en : llvm::enumerate(shape)) {
    unsigned dimSize = en.value();
    multiDim[en.index()] = i32_val(remained % dimSize);
    remained = remained / dimSize;
  }
  return multiDim;
}

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape) {
  unsigned rank = shape.size();
  assert(rank > 0);
  SmallVector<Value> multiDim(rank);
  Value remained = linear;
  for (auto &&en : llvm::enumerate(shape)) {
    Value dimSize = i32_val(en.value());
    multiDim[en.index()] = urem(remained, dimSize);
    remained = udiv(remained, dimSize);
  }
  return multiDim;
}

Value linearize(RewriterBase &rewriter, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape, ArrayRef<unsigned> order) {
  return linearize(rewriter, loc, applyPermutation(multiDim, order),
                   applyPermutation(shape, order));
}

Value linearize(RewriterBase &rewriter, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape) {
  auto rank = multiDim.size();
  Value linear = i32_val(0);
  if (rank > 0) {
    linear = multiDim.back();
    for (auto [dim, dimShape] :
         llvm::reverse(llvm::zip(multiDim.drop_back(), shape.drop_back()))) {
      Value dimSize = i32_val(dimShape);
      linear = add(mul(linear, dimSize), dim);
    }
  }
  return linear;
}

Value addStringToModule(Location loc, RewriterBase &rewriter, StringRef key,
                        StringRef content) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto ctx = moduleOp.getContext();
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (key + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));

  llvm::SmallString<64> contentStr(content);
  size_t contentSize = contentStr.size_in_bytes();
  auto globalType = LLVM::LLVMArrayType::get(i8_ty, contentSize);

  LLVM::GlobalOp global;
  {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    global = rewriter.create<LLVM::GlobalOp>(
        UnknownLoc::get(ctx), globalType,
        /*isConstant=*/true, LLVM::Linkage::Internal, stringConstName,
        rewriter.getStringAttr(contentStr));
  }

  Value zero = i32_val(0);
  Type globalPtrType = LLVM::LLVMPointerType::get(ctx, global.getAddrSpace());
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(
      UnknownLoc::get(ctx), globalPtrType, global.getSymName());
  Value stringStart =
      gep(ptr_ty(ctx), i8_ty, globalPtr, SmallVector<Value>({zero}));
  return stringStart;
}

SmallVector<Value> getMultiDimOffset(Attribute layout, Location loc,
                                     RewriterBase &rewriter,
                                     const TargetInfoBase &targetInfo,
                                     unsigned elemId, RankedTensorType type,
                                     ArrayRef<unsigned> multiDimCTAInRepId,
                                     ArrayRef<unsigned> shapePerCTATile) {
  auto shape = type.getShape();
  unsigned rank = shape.size();
  if (auto blockedLayout = dyn_cast<BlockedEncodingAttr>(layout)) {
    auto multiDimOffsetFirstElem = emitBaseIndexForLayout(
        loc, rewriter, targetInfo, blockedLayout, type, false);
    SmallVector<Value> multiDimOffset(rank);
    SmallVector<unsigned> multiDimElemId = getMultiDimIndex<unsigned>(
        elemId, getSizePerThread(layout), getOrder(layout));
    for (unsigned d = 0; d < rank; ++d) {
      multiDimOffset[d] =
          add(multiDimOffsetFirstElem[d],
              i32_val(multiDimCTAInRepId[d] * shapePerCTATile[d] +
                      multiDimElemId[d]));
    }
    return multiDimOffset;
  }
  if (auto sliceLayout = mlir::dyn_cast<SliceEncodingAttr>(layout)) {
    unsigned dim = sliceLayout.getDim();
    auto parentEncoding = sliceLayout.getParent();
    auto parentSizePerThread = getSizePerThread(parentEncoding);
    auto parentShape = sliceLayout.paddedShape(shape);
    auto parentTy = RankedTensorType::get(parentShape, type.getElementType(),
                                          parentEncoding);
    auto offsets = emitOffsetForLayout(layout, type);
    auto parentOffset = emitOffsetForLayout(parentEncoding, parentTy);
    SmallVector<int> idxs;
    for (SmallVector<unsigned> off : offsets) {
      off.insert(off.begin() + dim, 0);
      auto it = std::find(parentOffset.begin(), parentOffset.end(), off);
      idxs.push_back(std::distance(parentOffset.begin(), it));
    }
    auto multiDimOffsetParent = getMultiDimOffset(
        parentEncoding, loc, rewriter, targetInfo, idxs[elemId], parentTy,
        sliceLayout.paddedShape(multiDimCTAInRepId),
        sliceLayout.paddedShape(shapePerCTATile));
    SmallVector<Value> multiDimOffset(rank);
    for (unsigned d = 0; d < rank + 1; ++d) {
      if (d == dim)
        continue;
      unsigned slicedD = d < dim ? d : (d - 1);
      multiDimOffset[slicedD] = multiDimOffsetParent[d];
    }
    return multiDimOffset;
  }
  if (auto mmaLayout = mlir::dyn_cast<NvidiaMmaEncodingAttr>(layout)) {
    assert(rank == 2 ||
           (rank == 3 && mmaLayout.isAmpere()) && "Unexpected rank");
    auto shapePerCTA = getShapePerCTA(mmaLayout, shape);
    auto instrShape = mmaLayout.getInstrShape();
    SmallVector<Value> mmaColIdx(2);
    SmallVector<Value> mmaRowIdx(2);
    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(32);
    Value laneId = urem(threadId, warpSize);
    Value warpId = udiv(threadId, warpSize);
    // TODO: fix the bug in MMAEncodingAttr document
    SmallVector<Value> multiDimWarpId(2);
    auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
    auto warpOrder = triton::gpu::getWarpOrder(mmaLayout);
    multiDimWarpId = delinearize(rewriter, loc, warpId, warpsPerCTA, warpOrder);
    Value _1 = i32_val(1);
    Value _2 = i32_val(2);
    Value _4 = i32_val(4);
    Value _8 = i32_val(8);
    Value _16 = i32_val(16);
    if (mmaLayout.isAmpere() || mmaLayout.isHopper()) {
      multiDimWarpId[rank - 1] = urem(
          multiDimWarpId[rank - 1],
          i32_val(ceil<unsigned>(shapePerCTA[rank - 1], instrShape[rank - 1])));
      multiDimWarpId[rank - 2] = urem(
          multiDimWarpId[rank - 2],
          i32_val(ceil<unsigned>(shapePerCTA[rank - 2], instrShape[rank - 2])));

      Value mmaGrpId = udiv(laneId, _4);
      Value mmaGrpIdP8 = add(mmaGrpId, _8);
      Value mmaThreadIdInGrp = urem(laneId, _4);
      Value mmaThreadIdInGrpM2 = mul(mmaThreadIdInGrp, _2);
      Value mmaThreadIdInGrpM2P1 = add(mmaThreadIdInGrpM2, _1);
      Value rowWarpOffset =
          mul(multiDimWarpId[rank - 2], i32_val(instrShape[rank - 2]));
      mmaRowIdx[0] = add(mmaGrpId, rowWarpOffset);
      mmaRowIdx[1] = add(mmaGrpIdP8, rowWarpOffset);
      Value colWarpOffset =
          mul(multiDimWarpId[rank - 1], i32_val(instrShape[rank - 1]));
      mmaColIdx[0] = add(mmaThreadIdInGrpM2, colWarpOffset);
      mmaColIdx[1] = add(mmaThreadIdInGrpM2P1, colWarpOffset);
    } else {
      llvm_unreachable("Unexpected MMALayout version");
    }

    SmallVector<Value> multiDimOffset(rank);
    if (mmaLayout.isHopper()) {
      unsigned elemIdRem4 = elemId % 4;
      unsigned nGrpId = elemId / 4;
      multiDimOffset[0] = elemIdRem4 < 2 ? mmaRowIdx[0] : mmaRowIdx[1];
      multiDimOffset[1] = elemIdRem4 % 2 == 0 ? mmaColIdx[0] : mmaColIdx[1];
      multiDimOffset[1] = add(multiDimOffset[1], i32_val(8 * nGrpId));
      multiDimOffset[0] = add(multiDimOffset[0], i32_val(multiDimCTAInRepId[0] *
                                                         shapePerCTATile[0]));
      multiDimOffset[1] = add(multiDimOffset[1], i32_val(multiDimCTAInRepId[1] *
                                                         shapePerCTATile[1]));
    } else if (mmaLayout.isAmpere()) {
      if (rank == 3)
        multiDimOffset[0] =
            add(multiDimWarpId[0],
                i32_val(multiDimCTAInRepId[0] * shapePerCTATile[0]));
      multiDimOffset[rank - 2] = elemId < 2 ? mmaRowIdx[0] : mmaRowIdx[1];
      multiDimOffset[rank - 1] = elemId % 2 == 0 ? mmaColIdx[0] : mmaColIdx[1];
      multiDimOffset[rank - 2] =
          add(multiDimOffset[rank - 2], i32_val(multiDimCTAInRepId[rank - 2] *
                                                shapePerCTATile[rank - 2]));
      multiDimOffset[rank - 1] =
          add(multiDimOffset[rank - 1], i32_val(multiDimCTAInRepId[rank - 1] *
                                                shapePerCTATile[rank - 1]));
    } else {
      llvm_unreachable("Unexpected MMALayout version");
    }
    return multiDimOffset;
  }
  if (isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>(layout)) {
    auto multiDimBase =
        emitBaseIndexForLayout(loc, rewriter, targetInfo, layout, type, false);
    SmallVector<SmallVector<unsigned>> offsets;
    assert(rank == 2);
    SmallVector<Value> multiDimOffset(rank);
    if (auto mfmaLayout = dyn_cast<AMDMfmaEncodingAttr>(layout)) {
      emitMfmaOffsetForCTA(mfmaLayout, offsets, 0, multiDimCTAInRepId[0],
                           multiDimCTAInRepId[1]);
    } else if (auto wmmaLayout = dyn_cast<AMDWmmaEncodingAttr>(layout)) {
      emitWmmaOffsetForCTA(wmmaLayout, offsets, 0, multiDimCTAInRepId[0],
                           multiDimCTAInRepId[1]);
    }
    multiDimOffset[0] = add(multiDimBase[0], i32_val(offsets[elemId][0]));
    multiDimOffset[1] = add(multiDimBase[1], i32_val(offsets[elemId][1]));
    return multiDimOffset;
  }
  llvm_unreachable("unexpected layout in getMultiDimOffset");
}

SmallVector<Value> getWrappedMultiDimOffset(
    RewriterBase &rewriter, Location loc, ArrayRef<Value> multiDimOffset,
    ArrayRef<unsigned> shape, SmallVector<unsigned> shapePerCTATile,
    SmallVector<int64_t> shapePerCTA) {
  unsigned rank = shape.size();
  SmallVector<Value> multiDimOffsetWrapped(rank);
  for (unsigned d = 0; d < rank; ++d) {
    if (shapePerCTATile[d] > shapePerCTA[d])
      multiDimOffsetWrapped[d] = urem(multiDimOffset[d], i32_val(shape[d]));
    else
      multiDimOffsetWrapped[d] = multiDimOffset[d];
  }
  return multiDimOffsetWrapped;
}

SmallVector<Value> convertMxfp4x2ToBf16x2(RewriterBase &rewriter, Location loc,
                                          ArrayRef<Value> values) {
  SmallVector<Value> results;
  for (auto v : values) {
    auto em0 = and_(v, i8_val(0x70));
    auto em1 = and_(v, i8_val(0x7));
    Value v0 = or_(shl(zext(i16_ty, em0), i16_val(2)),
                   shl(zext(i16_ty, and_(v, i8_val(0x80))), i16_val(8)));
    Value v1 = or_(shl(zext(i16_ty, em1), i16_val(6)),
                   shl(zext(i16_ty, and_(v, i8_val(0x8))), i16_val(12)));

    // Three cases:
    // 1) x is normal and non-zero: Correct bias
    v0 = select(icmp_ne(and_(em0, i8_val(0x60)), i8_val(0)),
                add(v0, i16_val((127 - 1) << 7)), v0);
    v1 = select(icmp_ne(and_(em1, i8_val(0x6)), i8_val(0)),
                add(v1, i16_val((127 - 1) << 7)), v1);

    // 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in
    // bf16
    v0 = bitcast(select(icmp_eq(em0, i8_val(0x10)),
                        or_(i16_val(16128), and_(v0, i16_val(0x8000))), v0),
                 bf16_ty);
    v1 = bitcast(select(icmp_eq(em1, i8_val(0x1)),
                        or_(i16_val(16128), and_(v1, i16_val(0x8000))), v1),
                 bf16_ty);
    // 3) x is zero, nothing to do
    results.push_back(v0);
    results.push_back(v1);
  }
  return results;
}

Value mxfpScaleBf16(RewriterBase &rewriter, Location loc, Value v,
                    Value scale) {
  Value vBf16 = bitcast(v, bf16_ty);
  Value nanBf16 = bitcast(i16_val(0x7fff), bf16_ty);
  Value scaleIsNan = icmp_eq(scale, i8_val(0xff));
  Value scaleBf16 = bitcast(shl(zext(i16_ty, scale), i16_val(7)), bf16_ty);
  Value scaledBf16 = fmul(vBf16, scaleBf16);
  // Account for NaN in the scale as per the mxfp specification.
  return select(scaleIsNan, nanBf16, scaledBf16);
};

} // namespace LLVM
} // namespace mlir
