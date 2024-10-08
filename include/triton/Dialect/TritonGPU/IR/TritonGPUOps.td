#ifndef TRITONGPU_OPS
#define TRITONGPU_OPS

include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

class TTG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonGPU_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

class TTG_Type<string name, string typeMnemonic,
        list<Trait> traits = []> : TypeDef<TritonGPU_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TTG_AsyncToken : TTG_Type<"AsyncToken",
                                    "async.token", []> {
  let summary = "async token type";
  let description = [{
    `ttg.async.token` is a type returned by an asynchronous operation.
    It is used to establish an SSA-based link between async operations
    and operations that group or synchronize the async operations.
  }];
}


def TTG_ConvertLayoutOp : TTG_Op<"convert_layout",
                                 [SameOperandsAndResultShape,
                                  SameOperandsAndResultElementType,
                                  Pure]> {
  let summary = "convert layout";

  let arguments = (ins TT_Tensor:$src);

  let results = (outs TT_Tensor:$result);

  let hasCanonicalizer = 1;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def TTG_AsyncWaitOp : TTG_Op<"async_wait"> {
  let summary = "async wait";

  let arguments = (ins Variadic<TTG_AsyncToken>:$asyncToken, I32Attr:$num);

  let assemblyFormat = "$asyncToken attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}

def TTG_AsyncBulkWaitOp : TTG_Op<"async_bulk_wait"> {
  let summary = "async bulk wait";

  let arguments = (ins I32Attr:$num);

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 90;
    }
  }];
}

def TTG_AsyncCommitGroupOp : TTG_Op<"async_commit_group"> {
  let summary = "async commit group";

  let results = (outs TTG_AsyncToken:$asyncToken);
  let arguments = (ins Variadic<TTG_AsyncToken>:$inputTokens);

  let assemblyFormat = [{
    $inputTokens attr-dict
  }];

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 80;
    }
  }];
}

def TTG_AsyncBulkCommitGroupOp : TTG_Op<"async_bulk_commit_group"> {
  let summary = "async bulk commit group";

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 90;
    }
  }];
}

def TTG_AsyncCopyGlobalToLocalOp : TTG_Op<"async_copy_global_to_local",
                                    [AttrSizedOperandSegments,
                                     MemoryEffects<[MemRead<GlobalMemory>, MemWrite<SharedMemory>]>,
                                     TypesMatchWith<"infer mask type from src type",
                                                    "src", "mask", "getI1SameShape($_self)",
                                                    "($_op.getOperands().size() <= 3) || std::equal_to<>()">,
                                     TypesMatchWith<"infer other type from src type",
                                                    "src", "other", "getPointeeType($_self)",
                                                    "($_op.getOperands().size() <= 4) || std::equal_to<>()">]> {
  let summary = "copy data from global memory to local memory asynchronously";

  let description = [{
    This operation copies data from global memory to local memory asynchronously.
    This is analogue to tt.load except the data are copied to local memory pointed
    by by the memory descriptor instread of a distributed tensor. The rest of the
    operands are the same as tt.load.
  }];

  let arguments = (ins TT_PtrTensor:$src, TT_MemDescType:$dst,
                       Optional<I1Tensor>:$mask, Optional<TT_Type>:$other,
                       TT_CacheModifierAttr:$cache, TT_EvictionPolicyAttr:$evict,
                       BoolAttr:$isVolatile);

  let builders = [
      OpBuilder<(ins "Value":$src, "Value":$dst,
                     "triton::CacheModifier":$cache,
                     "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
  ];

  let results = (outs TTG_AsyncToken:$token);

  //let assemblyFormat = [{
  //  $src `,` $dst ``
  //  $index, $mask, $other
  //  attr-dict `:` type($src) `->` type($dst)
  //}];

  let extraClassDeclaration = [{
    static DenseSet<unsigned> getEligibleLoadByteWidth(int computeCapability) {
      DenseSet<unsigned> validLoadBytes;
      if (computeCapability >= 80) {
        validLoadBytes = {4, 8, 16};
      }
      return validLoadBytes;
    }
  }];

  let assemblyFormat = [{
    $src `,` $dst (`mask` $mask^)? (`other` $other^)? attr-dict `:` type($src) `->` type($dst)
  }];
}


// Allocate shared memory
def TTG_LocalAllocOp : TTG_Op<"local_alloc", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "allocate tensor";
  let description = [{
    This operation allocates buffer in shared memory and return a descriptor
    containing the address and a view of the buffer.
  }];
  let arguments = (ins Optional<TT_Tensor>:$init);

  let assemblyFormat = [{$init attr-dict `:` functional-type(operands, results)}];

  let results = (outs TT_MemDescType:$result);
}

// Deallocate shared memory
def TTG_LocalDeallocOp : TTG_Op<"local_dealloc", [MemoryEffects<[MemFree<SharedMemory>]>]> {
  let summary = "dealloc buffer";

  let description = [{
    This operation deallocate a buffer explicitly. Using the buffer after this operation is undefined.
  }];

  let arguments = (ins TT_MemDescType:$ptr);

  let assemblyFormat = [{$ptr attr-dict `:` qualified(type($ptr))}];
}

def TTG_MemDescSubviewOp : TTG_Op<"memdesc_subview", [Pure]> {
  let summary = "take a subview of the descriptor.";

  let description = [{
    This operation returns a new descriptor representing a subview of the buffer.
    It doesn't affect the underlying memory. The subview can be rank-reduced.
  }];
  let arguments = (
    ins TT_MemDescType:$desc, Variadic<I32>:$offsets);

  let assemblyFormat = [{$desc `[` $offsets `]` attr-dict `:` qualified(type($desc)) `->` qualified(type($result))}];

  let results = (outs TT_MemDescType:$result);
}

def TTG_LocalLoadOp : TTG_Op<"local_load", [MemoryEffects<[MemRead<SharedMemory>]>]> {
  let summary = "Load a buffer from local memory into a distributed tensor";

  let description = [{
    Load a tensor from the local memory descriptor into a distributed tensor.
  }];
  let arguments = (ins TT_MemDescType:$src);
  let assemblyFormat = [{$src attr-dict `:` qualified(type($src)) `->` type($result)}];
  let results = (outs TT_Tensor:$result);
}

#endif
