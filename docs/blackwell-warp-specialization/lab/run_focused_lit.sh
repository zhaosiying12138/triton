#!/usr/bin/env bash
set -euo pipefail

TRITON_SNAPSHOT=bf64a5db1bc8aab0fd4f0076e60f6c367852e47d
WS_BUILD_ROOT=${WS_BUILD_ROOT:-/home/zhaosiying/builds/triton-ws-bf64a5db}
WS_JOBS=${WS_JOBS:-24}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
TRITON_BUILD=${WS_TRITON_BUILD:-"${WS_BUILD_ROOT}/triton-build"}
LLVM_BUILD=${WS_LLVM_BUILD:-"${WS_BUILD_ROOT}/llvm-build-gcc15"}
VENV=${WS_VENV:-"${WS_BUILD_ROOT}/venv"}
LIT_BIN="${VENV}/bin/lit"
TRITON_OPT="${TRITON_BUILD}/bin/triton-opt"
FILECHECK_BIN="${LLVM_BUILD}/bin/FileCheck"
NOT_BIN="${LLVM_BUILD}/bin/not"
LIT_SITE_CFG="${TRITON_BUILD}/test/lit.site.cfg.py"
LOG_PATH=${WS_FOCUSED_LIT_LOG:-"${SCRIPT_DIR}/../evidence/raw/focused-lit.log"}

die() {
  echo "error: $*" >&2
  exit 1
}

require_executable() {
  [[ -x "$1" ]] || die "required executable not found: $1"
}

[[ "${WS_JOBS}" =~ ^[1-9][0-9]*$ ]] ||
  die "WS_JOBS must be a positive integer, got: ${WS_JOBS}"

git -C "${REPO_ROOT}" cat-file -e "${TRITON_SNAPSHOT}^{commit}" 2>/dev/null ||
  die "frozen Triton snapshot is unavailable: ${TRITON_SNAPSHOT}"
git -C "${REPO_ROOT}" merge-base --is-ancestor "${TRITON_SNAPSHOT}" HEAD ||
  die "current checkout does not descend from ${TRITON_SNAPSHOT}"

require_executable "${LIT_BIN}"
require_executable "${TRITON_OPT}"
require_executable "${FILECHECK_BIN}"
require_executable "${NOT_BIN}"
[[ -f "${LIT_SITE_CFG}" ]] ||
  die "generated Triton lit site config not found: ${LIT_SITE_CFG}"

LIT_VERSION=$("${LIT_BIN}" --version)
[[ "${LIT_VERSION}" == "lit 18."* ]] ||
  die "expected the venv lit 18 runner, got: ${LIT_VERSION}"

# Keep this list explicit: lit must receive build-tree-relative pseudo-paths so
# the generated lit.site.cfg.py can map them back to the frozen source tree.
WS_LIT_TESTS=(
  test/TritonGPU/automatic-warp-specialization.mlir
  test/TritonGPU/partition-scheduling.mlir
  test/TritonGPU/partition-loops.mlir
  test/NVWS/lower_warp_group.mlir
  test/NVWS/insert_aref.mlir
  test/NVWS/aref-tmem-insertion.mlir
  test/NVWS/lower_aref.mlir
  test/NVWS/hoist_tmem_store.mlir
  test/TritonGPU/optimize-partition-warps.mlir
  test/Conversion/allocate_warp_groups.mlir
  test/Conversion/warp_specialize_to_llvm.mlir
  test/Conversion/tritongpu_to_llvm_blackwell.mlir
  test/Conversion/lower_tensor_memory_to_llvm.mlir
  test/TritonNvidiaGPU/membar-cluster.mlir
  test/TritonNvidiaGPU/cluster-barrier-mbar-allocator.mlir
  test/TritonNvidiaGPU/tmem_barrier_insertion.mlir
  test/Conversion/tritonnvidiagpu_to_llvm.mlir
  test/TritonGPU/partition-verifier-locality.mlir
  test/NVWS/invalid.mlir
  test/TritonGPU/proxy_fence_insertion.mlir
  test/TritonNvidiaGPU/test_tensor_memory_allocation.mlir
)

for test_path in "${WS_LIT_TESTS[@]}"; do
  [[ -f "${REPO_ROOT}/${test_path}" ]] ||
    die "focused lit test is missing: ${test_path}"
done

mkdir -p "$(dirname -- "${LOG_PATH}")"

{
  printf 'Triton snapshot: %s\n' "${TRITON_SNAPSHOT}"
  printf 'Triton checkout: %s\n' "${REPO_ROOT}"
  printf 'Triton HEAD: %s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  printf 'Triton build: %s\n' "${TRITON_BUILD}"
  printf 'LLVM build: %s\n' "${LLVM_BUILD}"
  printf 'lit: %s (%s)\n' "${LIT_BIN}" "${LIT_VERSION}"
  printf 'triton-opt: %s\n' "${TRITON_OPT}"
  printf 'FileCheck: %s\n' "${FILECHECK_BIN}"
  "${FILECHECK_BIN}" --version
  printf 'not: %s\n' "${NOT_BIN}"
  printf 'workers: %s\n' "${WS_JOBS}"
  printf 'test files: %s\n\n' "${#WS_LIT_TESTS[@]}"

  (
    cd "${TRITON_BUILD}"
    env PATH="${TRITON_BUILD}/bin:${LLVM_BUILD}/bin:${VENV}/bin:${PATH}" \
      "${LIT_BIN}" \
        -j "${WS_JOBS}" \
        -sv \
        --show-pass \
        --no-progress-bar \
        --time-tests \
        --timeout 300 \
        "-Dfilecheck=${FILECHECK_BIN}" \
        "${WS_LIT_TESTS[@]}"
  )
} 2>&1 | tee "${LOG_PATH}"
