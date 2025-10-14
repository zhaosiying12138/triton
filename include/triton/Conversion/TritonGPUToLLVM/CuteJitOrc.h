#pragma once
#include <string>

namespace mlir::triton {
namespace runtime {

struct JitStatus {
  bool ok;
  std::string msg;
};

/// 运行期：
///   - 生成 C++ 源，模板参数用 warpsPerCTA
///   - 调 clang++ 产出 .bc
///   - 用 ORC LLJIT JIT 并执行 cute_jit_entry()
JitStatus initCuteJitORC(int sizePerThread, int threadsPerWarp,
                              int warpsPerCTA, int tensorShapeSize);

int getCuteLayoutSize();

JitStatus calculateCuteLayout(int *data);

} // namespace runtime
} // namespace triton
