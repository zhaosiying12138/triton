# Warp-specialization evidence test plan

This file records the completed compile-only checks for the matching local
Triton and LLVM source builds. No runtime or performance measurement is
inferred from these results.

## A. Safety and fixture checks

- [x] `python3 -m py_compile ws_study.py cases/tma_matmul.py`
- [x] `python3 ws_study.py validate` passes static manifest/source checks.
- [x] `triton-opt` parses both `ws_skeleton` TTGIR fixtures.
- [x] `tritongpu-allocate-warp-groups` on the WS fixture produces
  `ttg.total-num-warps`, `warpGroupStartIds`, `requestedRegisters`, and
  `actualRegisters`.
- [x] a non-SM103 `compile --target` value is rejected before Triton import.
- [x] an inherited `TRITON_OVERRIDE_ARCH` is cleared; only
  `GPUTarget("cuda", 103, 32)` selects the compile target.
- [x] `_compile_case` and `_evaluate_record` both reject a retargeted IR
  fixture, so trace/render cannot bypass the SM103 source lock.
- [x] Python fixtures require an exact reviewed path/entry/SHA-256 tuple;
  manifest membership and AST screening alone cannot authorize execution.
- [x] import/compile probes use only in-tree backends and clear inherited
  backend/pass plugins and code-generation switches.
- [x] `build/raw` and `build/curated` are ignored; top-level `../evidence` is
  versionable.

## B. Focused compiler regression suite

After the source-built LLVM and Triton tools are ready, run the 31-file focused
lit suite. It covers AutomaticWS, latency assignment/schedule/lower-loop,
partition scheduling and loop materialization, explicit WS verifier negatives,
the complete `test/NVWS` directory including dialect syntax and stage/phase
assignment, warp-group/ARef/TMEM-ARef lowering, warp-count optimization and allocation,
LLVM conversion, proxy fences, ConSan, cluster synchronization, and TMEM
allocation and hazard barriers.

```bash
./run_focused_lit.sh
```

Defaults and overrides:

- `WS_BUILD_ROOT=/home/zhaosiying/builds/triton-ws-bf64a5db`;
- `WS_JOBS=24`;
- `WS_FOCUSED_LIT_LOG=../evidence/raw/focused-lit.log`, resolved relative to
  this lab by default;
- `WS_TRITON_BUILD`, `WS_LLVM_BUILD`, and `WS_VENV` may override individual
  directories below the build root.

The runner deliberately combines venv lit 18 with the pinned LLVM build's
`FileCheck` and `not`. Passing source-tree absolute test paths or using the
pinned LLVM's lit 23 runner bypasses or conflicts with this frozen Triton lit
configuration. No selected file has a `REQUIRES`, `UNSUPPORTED`, or `XFAIL`
gate, and no selected `RUN:` line invokes a GPU, CUDA driver, or `ptxas`.

Completed log review:

- [x] the provenance header names the frozen Triton snapshot and intended
  Triton/LLVM build directories;
- [x] lit reports all 31 test files, representing 43 `RUN:` lines;
- [x] every test passes, including verifier-negative cases using LLVM `not`;
- [x] no tool-resolution warning points outside the pinned build root;
- [x] `../evidence/raw/focused-lit.log` contains no kernel-runtime or performance
  claim.

## C. SM103 compile-only run after the native build finishes

Run from this directory. These commands assemble cubins as data but never load
or launch them.

```bash
python3 ws_study.py inspect

for case in ws_skeleton tma_matmul; do
  python3 ws_study.py compile --case "$case" --target sm103 --ws off
  python3 ws_study.py compile --case "$case" --target sm103 --ws on
done

python3 ws_study.py trace --case tma_matmul --ws both --passes all
python3 ws_study.py trace --case ws_skeleton --ws both --passes all

python3 ws_study.py disassemble --case ws_skeleton
python3 ws_study.py disassemble --case tma_matmul
python3 ws_study.py validate --require-artifacts --show-optional
python3 ws_study.py render
```

Completed artifact review:

- [x] every `compile.json` says `compiled-not-launched` and records
  `cuda:103:32` in both the locked target and compiler metadata;
- [x] every PTX file declares `.target sm_103a` and no other SM target;
- [x] all hashes validate;
- [x] all four records bind a full pass trace and the reviewed harness,
  source, manifest, native extension, assembler, and environment contract;
- [x] `tma_matmul/on` contains the required WS, TMA, TCGen05, TMEM, and
  mbarrier families;
- [x] post-run ISA audit records that the TCGen05 family only asserts
  `mma+commit`: the frozen source/tests/PTX omit the canonical
  `tcgen05.fence::after_thread_sync`, so cross-role ordering and runtime
  correctness remain unqualified despite structural PASS;
- [x] `tma_matmul/off` retains the common data/matmul path but contains no
  `ttg.warp_specialize`, ARef channel, WS cluster-barrier protocol, or dynamic
  register redistribution;
- [x] `ws_skeleton/on` lowers to named-barrier dispatch and `setmaxnreg`, while
  the off fixture contains neither;
- [x] optional-family misses were inspected; no unsupported optional family was
  promoted to required in `manifest.json`;
- [x] all four disassembly records report `disassembled-not-executed` and the
  locked `cuda:103:32` target;
- [x] `build/curated/evidence.{json,md}` and `build/curated/index.html` quote
  only validated artifact lines and make no runtime claim.

Publication was run only after the checks above completed:

```bash
python3 ws_study.py render --publish
```

- [x] `evidence/sm103-compile-only/{evidence.json,evidence.md,index.html}` was
  published from the validated canonical four-variant view.

If committed evidence already differs, inspect the diff first. Use
`--publish --force` only after deliberately accepting replacement.

## D. Deferred exact-SM103 device qualification

These items are deliberately recorded but **not run**. They require an actual
SM103 device and a separate authorization window; an SM120 result cannot fill
any box below.

- [ ] Resolve or obtain an authoritative disposition for the missing
  `tcgen05.fence::after_thread_sync` before treating cross-role MMA→TMEM-load
  execution as qualified. Preserve the exact PTX/SASS sequence used by each
  runtime binary.
- [ ] Run WS on/off numerical comparisons against an independent reference for
  canonical GEMM first, then persistent GEMM, attention, grouped GEMM, scaled
  MMA, and the selected 2CTA cases. Include boundary shapes and repeated runs
  capable of exposing intermittent ordering or deadlock failures.
- [ ] Add watchdog-bounded stress runs for empty/short/full K loops, persistent
  tile tails, descriptor changes, multi-consumer ARefs, accumulator ping-pong,
  early worker retirement, and cluster exit/phase rollover.
- [ ] Compare on/off latency distributions only after controlling input,
  `num_warps`, `num_stages`, launch geometry, clocks, warmup, cache state,
  toolchain, and compilation options. Report median and tail distributions,
  never a single best timing.
- [ ] Capture achieved occupancy, registers/thread, shared memory, active
  warps, eligible warps/cycle, tensor-pipe utilization, TMA throughput,
  barrier stalls, scoreboard stalls, and instruction issue by role. Correlate
  counters with the already preserved pass trace and SASS rather than inferring
  efficiency from static `requestedRegisters`.
- [ ] Qualify 2CTA launch legality, cluster residency, peer progress and
  slot/parity rollover separately from the one-CTA study. Do not merge its
  numbers into the canonical GEMM conclusion.
- [ ] Publish raw commands, device/driver/tool versions, clocks, environment,
  hashes, failures and negative results alongside any chart. Until then the
  prohibited claims remain: “WS is faster”, “WS improves occupancy”, “this
  partition count is optimal”, or “another target proves SM103 behavior”.

## Blocker status

- [x] The original Python-package/`libtriton.so` revision mismatch was resolved
  by rebuilding Triton against the locked LLVM source build.
- [x] The first Triton native build exposed a missing LLVM AMDGPU target in
  `llvm.cc`; reconfiguring LLVM from `Native;NVPTX` to
  `Native;NVPTX;AMDGPU` completed the 387/387 LLVM increment and the following
  8/8-target Triton increment.
- [x] The focused lit suite, four compile records, four full pass traces, four
  disassemblies, and artifact validation now pass.

There is no active build or compile-only validation blocker. The remaining
boundary is intentional: this project did not load or launch a cubin and makes
no SM103 runtime, occupancy, or performance claim.
