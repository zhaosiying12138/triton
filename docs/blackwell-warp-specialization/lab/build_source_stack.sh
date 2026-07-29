#!/usr/bin/env bash
set -euo pipefail

TRITON_SNAPSHOT=bf64a5db1bc8aab0fd4f0076e60f6c367852e47d
LLVM_SNAPSHOT=850a2b1b975c061ae0fc982ba68064d305485cb2

if [[ -z "${WS_BUILD_ROOT:-}" ]]; then
  echo "set WS_BUILD_ROOT to an external build directory" >&2
  echo "example: WS_BUILD_ROOT=/home/zhaosiying/builds/triton-ws-bf64a5db $0 all" >&2
  exit 2
fi

ACTION=${1:-all}
WS_JOBS=${WS_JOBS:-24}
REPO_ROOT=$(git rev-parse --show-toplevel)
LLVM_SOURCE=${LLVM_SOURCE:-"${WS_BUILD_ROOT}/llvm-project-src"}
LLVM_BUILD=${LLVM_BUILD:-"${WS_BUILD_ROOT}/llvm-build-gcc15"}
TRITON_BUILD=${TRITON_BUILD:-"${WS_BUILD_ROOT}/triton-build"}
TRITON_CACHE=${TRITON_CACHE:-"${WS_BUILD_ROOT}/triton-home"}
VENV=${VENV:-"${WS_BUILD_ROOT}/venv"}
CUDA_BIN=${CUDA_BIN:-/usr/local/cuda-13.3/bin}
JSON_SYSPATH=${JSON_SYSPATH:-/usr}

die() {
  echo "error: $*" >&2
  exit 1
}

require_tool() {
  command -v "$1" >/dev/null 2>&1 || die "required tool not found: $1"
}

check_triton_snapshot() {
  git -C "${REPO_ROOT}" cat-file -e "${TRITON_SNAPSHOT}^{commit}" 2>/dev/null ||
    die "Triton snapshot ${TRITON_SNAPSHOT} is not present"
  git -C "${REPO_ROOT}" merge-base --is-ancestor "${TRITON_SNAPSHOT}" HEAD ||
    die "current checkout is not based on frozen Triton snapshot ${TRITON_SNAPSHOT}"
}

check_llvm_lock() {
  local lockfile="${REPO_ROOT}/cmake/llvm-info.json"
  local actual
  actual=$(python3.14 -c 'import json,sys; print(json.load(open(sys.argv[1]))["llvm_hash"])' "${lockfile}")
  [[ "${actual}" == "${LLVM_SNAPSHOT}" ]] ||
    die "cmake/llvm-info.json locks ${actual}; script expects ${LLVM_SNAPSHOT}"
}

prepare_venv() {
  if [[ ! -x "${VENV}/bin/python" ]]; then
    python3.14 -m venv "${VENV}"
  fi
  "${VENV}/bin/python" -m pip install \
    'cmake<4' ninja setuptools wheel nanobind pytest \
    'lit==18.1.8' 'psutil==7.2.2'
}

prepare_llvm_source() {
  if [[ ! -d "${LLVM_SOURCE}/.git" ]]; then
    git clone --filter=blob:none --no-checkout \
      https://github.com/llvm/llvm-project.git "${LLVM_SOURCE}"
    git -C "${LLVM_SOURCE}" fetch --depth=1 origin "${LLVM_SNAPSHOT}"
    git -C "${LLVM_SOURCE}" checkout --detach "${LLVM_SNAPSHOT}"
  fi
  local actual
  actual=$(git -C "${LLVM_SOURCE}" rev-parse HEAD)
  [[ "${actual}" == "${LLVM_SNAPSHOT}" ]] ||
    die "LLVM source is ${actual}; expected ${LLVM_SNAPSHOT}. Refusing to rewrite it."
}

configure_llvm() {
  require_tool gcc-15
  require_tool g++-15
  require_tool ccache
  prepare_venv
  prepare_llvm_source
  # Triton's current Python binding links and initializes both in-tree
  # codegen backends even though this study emits NVIDIA artifacts only.
  "${VENV}/bin/cmake" -G Ninja \
    -S "${LLVM_SOURCE}/llvm" \
    -B "${LLVM_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-15 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS='mlir;llvm;lld;clang' \
    -DLLVM_TARGETS_TO_BUILD='Native;NVPTX;AMDGPU' \
    -DLLVM_USE_LINKER=lld \
    -DLLVM_PARALLEL_LINK_JOBS=2 \
    -DLLVM_ENABLE_PCH=OFF
}

build_llvm() {
  [[ -f "${LLVM_BUILD}/build.ninja" ]] || die "configure LLVM first"
  "${VENV}/bin/cmake" --build "${LLVM_BUILD}" --parallel "${WS_JOBS}"
}

build_triton() {
  [[ -x "${LLVM_BUILD}/bin/mlir-opt" ]] || die "complete the LLVM source build first"
  [[ -f "${JSON_SYSPATH}/include/nlohmann/json.hpp" ]] ||
    die "nlohmann/json headers are missing below ${JSON_SYSPATH}"
  for tool in ptxas cuobjdump nvdisasm; do
    [[ -x "${CUDA_BIN}/${tool}" ]] || die "CUDA tool is missing: ${CUDA_BIN}/${tool}"
  done
  prepare_venv
  (
    cd "${REPO_ROOT}"
    MAX_JOBS="${WS_JOBS}" \
    LLVM_INCLUDE_DIRS="${LLVM_BUILD}/include" \
    LLVM_LIBRARY_DIR="${LLVM_BUILD}/lib" \
    LLVM_SYSPATH="${LLVM_BUILD}" \
    JSON_SYSPATH="${JSON_SYSPATH}" \
    TRITON_BUILD_DIR="${TRITON_BUILD}" \
    TRITON_HOME="${TRITON_CACHE}" \
    TRITON_OFFLINE_BUILD=1 \
    TRITON_BUILD_PROTON=false \
    TRITON_BUILD_WITH_CLANG_LLD=true \
    TRITON_BUILD_WITH_CCACHE=true \
    TRITON_PTXAS_PATH="${CUDA_BIN}/ptxas" \
    TRITON_PTXAS_BLACKWELL_PATH="${CUDA_BIN}/ptxas" \
    TRITON_CUOBJDUMP_PATH="${CUDA_BIN}/cuobjdump" \
    TRITON_NVDISASM_PATH="${CUDA_BIN}/nvdisasm" \
      "${VENV}/bin/python" -m pip install -e . --no-build-isolation
  )
  "${VENV}/bin/python" -c \
    'import triton, triton._C.libtriton; print(triton.__version__)'
}

check_triton_snapshot
check_llvm_lock
case "${ACTION}" in
  configure-llvm)
    configure_llvm
    ;;
  build-llvm)
    prepare_venv
    build_llvm
    ;;
  build-triton)
    build_triton
    ;;
  all)
    configure_llvm
    build_llvm
    build_triton
    ;;
  *)
    die "unknown action ${ACTION}; use configure-llvm, build-llvm, build-triton, or all"
    ;;
esac
