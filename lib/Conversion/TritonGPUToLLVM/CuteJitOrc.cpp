#include "triton/Conversion/TritonGPUToLLVM/CuteJitOrc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>

#include <dlfcn.h>
#include <unistd.h>

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h" // DynamicLibrarySearchGenerator
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
// #include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
using namespace llvm::orc;

namespace mlir::triton {
namespace runtime {

static std::unique_ptr<LLJIT> J;
static LLVMContext Ctx;

static inline std::string shell_escape(const std::string& s) {
  std::string out; out.reserve(s.size() + 2);
  out.push_back('\'');
  for (char c : s) {
    if (c=='\'') out += "'\\''";
    else out.push_back(c);
  }
  out.push_back('\'');
  return out;
}

JitStatus initCuteJitORC(int sizePerThread, int threadsPerWarp,
                             int warpsPerCTA, int tensorShapeSize) {
  const std::string cudaInclude = "/usr/local/cuda-12.5/include";
  const std::string& cutlassInclude =
      "/home/zhaosiying/codebase/compiler/new_triton/cutedsl/cutlass/include";

  // 0) 初始化 ORC（一次即可，多次也安全）
  static bool inited = false;
  if (!inited) {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();
    inited = true;
  }

  // 1) 临时目录 + 路径
  char tmpTpl[] = "/home/zhaosiying/codebase/compiler/new_triton/triton/third_party/nvidia/backend/jit";
  std::string dir(tmpTpl);
  std::string src = dir + "/jit_cute.cpp";
  std::string bc  = dir + "/jit_cute_host.bc";
  std::string device_bc  = dir + "/../lib/libcutelayout.bc";
  std::string log = dir + "/build.log";

  // 3) 调 clang++ 只做前端，生成 bitcode
  // 说明：
  //  -emit-llvm -c：产出 .bc
  //  -I<cutlassInclude>：可见 cute/layout.hpp
  //  只用 cstdio，不用 iostream，避免 c++ 运行库依赖
  std::ostringstream cmd;
  cmd << "clang++ -std=c++17 -O2 -fPIC -c -emit-llvm "
      << "-I" << shell_escape(cutlassInclude) << " "
      << "-I" << shell_escape(cudaInclude) << " "
      << "-DsizePerThread_=" << sizePerThread << " "
      << "-DthreadsPerWarp_=" << threadsPerWarp << " "
      << "-DwarpsPerCTA_=" << warpsPerCTA << " "
      << "-DtensorShapeSize_=" << tensorShapeSize << " "
      << shell_escape(src) << " -o " << shell_escape(bc)
      << " 2> " << shell_escape(log);

  if (std::system(cmd.str().c_str()) != 0) {
    std::ifstream ifs(log); std::string errs((std::istreambuf_iterator<char>(ifs)), {});
    return {false, "clang failed:\n" + cmd.str() + "\n" + errs};
  }

  // compile device bc
  cmd.str("");
  cmd << "clang++ -x cuda -O3 -c -emit-llvm" << " "
      << "--cuda-gpu-arch=sm_86 --cuda-device-only" << " "
      << "-fcuda-flush-denormals-to-zero" << " "
      << "-I" << shell_escape(cutlassInclude) << " "
      << "-I" << shell_escape(cudaInclude) << " "
      << "-DsizePerThread_=" << sizePerThread << " "
      << "-DthreadsPerWarp_=" << threadsPerWarp << " "
      << "-DwarpsPerCTA_=" << warpsPerCTA << " "
      << "-DtensorShapeSize_=" << tensorShapeSize << " "
      << shell_escape(src) << " -o " << shell_escape(device_bc);

  if (std::system(cmd.str().c_str()) != 0) {
    std::ifstream ifs(log); std::string errs((std::istreambuf_iterator<char>(ifs)), {});
    return {false, "clang failed:\n" + cmd.str() + "\n" + errs};
  }

  // 4) 读取 bitcode -> Module

  SMDiagnostic Err;
  auto MBOrErr = MemoryBuffer::getFile(bc);
  if (!MBOrErr) return {false, "read .bc failed"};
  auto MOrErr = parseIR(MBOrErr->get()->getMemBufferRef(), Err, Ctx);
  if (!MOrErr) {
    std::string msg;
    raw_string_ostream rso(msg); Err.print("CuteJIT", rso);
    return {false, "parseIR failed:\n" + rso.str()};
  }
  std::unique_ptr<Module> M = std::move(MOrErr);

  // 5) 建立 LLJIT
  auto JOrErr = LLJITBuilder().create();
  if (!JOrErr) return {false, "LLJIT create failed: " + toString(JOrErr.takeError())};
  J = std::move(*JOrErr);

  // 让 JIT 能解析到当前进程里的符号（printf 等）
  // ... 拿到 LLJIT 实例 J 之后
  auto &JD = J->getMainJITDylib();

  // 1) 生成器：当前进程符号查找
  auto MO = J->getDataLayout().getGlobalPrefix();
  std::unique_ptr<DynamicLibrarySearchGenerator> Gen =
    cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(MO));

  // 2) 以 unique_ptr 形式 move 进去
  JD.addGenerator(std::move(Gen));

  // （可选）确保数据布局一致
  if (M->getDataLayout().isDefault())
    M->setDataLayout(J->getDataLayout());

  // 6) 加载模块并 JIT
  ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());
  // 迁移到独立 context（避免上面 Ctx 生命周期问题）
  std::unique_ptr<Module> M2 = CloneModule(*M);
  if (auto E = J->addIRModule(ThreadSafeModule(std::move(M2), std::move(TSCtx))))
    return {false, "addIRModule failed: " + toString(std::move(E))};

  return {true, "ok"};
}

int getCuteLayoutSize() {
  // 7) 查找并调用入口
  auto SymOrErr = J->lookup("get_cute_layout_size");
  if (!SymOrErr) return -1;
  using EntryFn = int (*)();
  EntryFn fn = SymOrErr->toPtr<EntryFn>();
  return fn();
}

JitStatus calculateCuteLayout(int *data) {
  // 7) 查找并调用入口
  auto SymOrErr = J->lookup("calculate_cute_layout");
  if (!SymOrErr) return {false, "calculateCuteLayout call failed"};
  using EntryFn = void (*)(int *);
  EntryFn fn = SymOrErr->toPtr<EntryFn>();
  fn(data);

  return {true, "ok"};
}

} // namespace runtime
} // namespace triton
