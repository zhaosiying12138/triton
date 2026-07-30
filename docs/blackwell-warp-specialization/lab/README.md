# SM103 compile-only warp-specialization lab

This lab records what the local Triton compiler transforms for NVIDIA SM103.
It is deliberately **compile-only**: it may ask `ptxas` to assemble PTX into a
cubin and may ask `cuobjdump`/`nvdisasm` to decode that cubin, but no command in
this directory loads or launches a cubin.

Completed compiler checks and the still-unrun exact-SM103 device qualification
checklist are kept in [TEST_PLAN.md](TEST_PLAN.md).

The target is locked in the CLI, manifest, source fixtures, compiler metadata,
and final PTX to `GPUTarget("cuda", 103, 32)` (`cuda:103:32`). Every compile
entry point repeats the source checks. Compilation uses in-tree backends only
and clears inherited backend/pass plugins, architecture overrides, and
code-generation switches before importing Triton and again after constructing
a Python AST source. Any other `--target` value is rejected before Triton is
imported.

## Requirements

There is no lab-specific `requirements.txt` and no `pip install` step. The lab
uses only Python's standard library plus the Triton checkout in which it lives.
For compilation, finish building this checkout and make the Blackwell `ptxas`
tool available through Triton's normal tool discovery. No CUDA device, PyTorch,
or tensor allocation is required.

Run commands from this directory or pass an absolute path to `ws_study.py`:

```bash
python3 ws_study.py inspect

python3 ws_study.py compile --case tma_matmul --target sm103 --ws off
python3 ws_study.py compile --case tma_matmul --target sm103 --ws on

python3 ws_study.py trace --case tma_matmul --passes all --ws both
python3 ws_study.py trace --case tma_matmul \
  --passes tritongpu-automatic-warp-specialization

python3 ws_study.py compile --case ws_skeleton --target sm103 --ws off
python3 ws_study.py compile --case ws_skeleton --target sm103 --ws on
python3 ws_study.py trace --case ws_skeleton --ws both --passes all
python3 ws_study.py disassemble --case ws_skeleton

python3 ws_study.py validate --require-artifacts
python3 ws_study.py render
python3 ws_study.py render --publish
```

To reproduce the matching native stack from source, use an external build
directory so the checkout stays clean:

```bash
WS_BUILD_ROOT=/home/zhaosiying/builds/triton-ws-bf64a5db \
  ./build_source_stack.sh all
```

The script verifies that this checkout descends from the frozen Triton commit,
checks out the exact LLVM revision from `cmake/llvm-info.json`, builds
LLVM/MLIR/Clang/LLD with the Native and NVPTX targets, and then builds this
Triton checkout against that build tree. It refuses to rewrite an existing LLVM
source tree whose `HEAD` differs from the lock.

After the native build finishes, run the focused compiler regression suite:

```bash
./run_focused_lit.sh
```

The script runs 31 checked-in MLIR files (43 `RUN:` lines) with 24 lit workers
by default. It uses the venv's lit 18 runner together with `triton-opt`,
`FileCheck`, and `not` from the pinned source builds. The LLVM checkout's own
lit 23 runner is intentionally not used: this frozen Triton test configuration
still constructs `ShTest` through the lit 18 interface. Tests are passed as
build-tree-relative `test/...` paths so the generated `lit.site.cfg.py` maps
them back to this source tree.

The combined provenance and test output is written to
`../evidence/raw/focused-lit.log`. Override the external build root, worker
count, or log destination when needed:

```bash
WS_BUILD_ROOT=/path/to/source-build \
WS_JOBS=24 \
WS_FOCUSED_LIT_LOG=/path/to/focused-lit.log \
  ./run_focused_lit.sh
```

This suite only invokes compiler tools. It does not need a GPU, CUDA driver,
`ptxas`, or a kernel launch. The raw log is review input; it is not promoted by
`ws_study.py render --publish` automatically.

`inspect` is safe before the native build is ready. It reports import and tool
readiness without initializing/querying a CUDA driver or querying a GPU; its
import probe uses the same in-tree-backend and sanitized-environment contract
as compilation.

## Output contract

Generated files are intentionally ignored:

```text
build/
  raw/<case>/<on|off>/
    ttir.mlir             # when the source is Triton Python
    ttgir.mlir
    llir.ll
    kernel.ptx
    kernel.cubin          # data only; never loaded by this lab
    compile.json          # provenance and SHA-256 hashes
    validation.json
    trace/full.mlir.log   # pass-manager dump, when requested
    trace/pass-index.json
    kernel.sass           # when disassemble succeeds
  curated/                 # ignored, reviewable publication staging area
    evidence.json
    evidence.md
    index.html
```

`raw` is an audit trail. `curated` is a separate, derived view containing a
small family matrix and source excerpts suitable for linking from the learning
document. Both are ignored build products. `render` never invents a missing
result: it only summarizes records that already exist under `build/raw`.

After reviewing `build/curated`, `render --publish` copies the canonical,
unfiltered view to the versionable
`../evidence/sm103-compile-only/` directory. Publication is deterministic and
will not overwrite different existing evidence unless `--force` is given;
identical files are left untouched. Publication requires both variants of every
manifest case, a hash-bound full pass trace for every variant, and successful
structural validation. Filtered renders remain staging-only so a partial case
cannot accidentally replace the canonical evidence set.

## Cases

- `tma_matmul` is adapted from
  `python/test/unit/language/test_warp_specialization.py`. Its on/off variants
  differ only in the `tl.range(..., warp_specialize=...)` constexpr. It is the
  end-to-end automatic-WS case for TMA, TCGen05, TMEM, and mbarrier.
- `ws_skeleton` is a lit-style TTGIR fixture. It intentionally has no matrix
  operations, allowing named-barrier dispatch and dynamic register
  redistribution to be studied without conflating them with TMA or TCGen05.

The manifest classifies each feature family as `required`, `optional`, or
`forbidden` for each case/variant. Cluster barriers are represented in the
schema but optional in these one-CTA starter cases; a future 2CTA case can make
that family required without changing the CLI.

## What the evidence proves

Successful compilation and validation prove that this checkout accepted the
SM103 target and produced the asserted IR/PTX structure. They do **not** prove
runtime correctness, deadlock freedom on silicon, occupancy, or performance.
Those claims are outside this lab's scope.

In particular, the `tcgen05` family asserts MMA issue plus completion commit,
while `tmem` separately asserts the co-presence of allocation, allocation-permit
release, `tcgen05.ld` plus `tcgen05.wait::ld`, and deallocation. These line-wise
patterns do not by themselves prove dynamic instruction order.
Neither family asserts a complete PTX ISA cross-thread ordering sequence. The
recorded artifacts omit `tcgen05.fence::after_thread_sync` before the cross-role
TMEM load; that known gap is documented in `../evidence/README.md` and must not
be hidden by the otherwise successful validation result.

## Adding a case

1. Add either a reviewed, device-free Python module exposing
   `make_source(bool)`, or on/off IR files under `cases/`.
2. Add the source, compile options, and per-variant assertions to
   `manifest.json`.
3. For a Python case, audit it and add its exact path, entry point, and SHA-256
   to `TRUSTED_PYTHON_CASES`; the manifest alone cannot authorize execution.
4. Prefer adapting a checked-in Triton test or lit fixture and document the
   provenance.
5. Run `inspect`, compile and trace both variants, then `validate` and
   `render`.

The reviewed digest allowlist is the primary trust boundary for Python cases;
the AST screen is a second check, not a general Python sandbox. It rejects
non-Triton imports, executable module-level statements, direct launch syntax,
dynamic evaluation, and known driver/load/run calls. The lab driver itself
only calls `triton.compile` and reads `CompiledKernel.asm`; it never indexes a
compiled kernel, accesses `CompiledKernel.run`, or calls a binary loader.

`inspect` searches the checkout, `PATH` (including a normal CUDA Toolkit such
as `/usr/local/cuda-13.3/bin`), and the explicit
`TRITON_PTXAS_BLACKWELL_PATH`, `TRITON_CUOBJDUMP_PATH`, and
`TRITON_NVDISASM_PATH` overrides. It reports the resolved tools without using a
CUDA driver.
