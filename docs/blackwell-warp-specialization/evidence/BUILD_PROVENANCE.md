# Source-build and validation provenance

This record describes the compiler stack used for the SM103 compile-only
warp-specialization evidence. It intentionally records no build duration,
kernel runtime, throughput, or speedup.

## Locked sources

| Component | Revision |
|---|---|
| Triton | `bf64a5db1bc8aab0fd4f0076e60f6c367852e47d` |
| LLVM project | `850a2b1b975c061ae0fc982ba68064d305485cb2` |

The Triton revision is the frozen upstream base of the study branch. The LLVM
revision is the value locked by that Triton tree's `cmake/llvm-info.json`.

## Final build configuration

| Property | Value |
|---|---|
| LLVM build type | `Release` |
| LLVM assertions | enabled |
| LLVM C/C++ compiler | GCC 15 (`gcc-15` / `g++-15`) |
| Triton C/C++ compiler | Clang 21.1.8 |
| Linker | `lld` |
| LLVM targets | `Native;NVPTX;AMDGPU` |
| Parallel build setting | `-j24` |
| Triton build | external CMake/Ninja build linked against the locked LLVM build |

The Triton native extension was built from a study-branch descendant whose
compiler tree is byte-for-byte unchanged outside this documentation package;
the frozen upstream base above remains the compiler-source identity checked by
the harness. The final LLVM target set matters even though the study compiles NVIDIA code.
The first full LLVM build, configured with `Native;NVPTX`, completed all
3010/3010 build steps. Triton's subsequent native build stopped in `llvm.cc`
because the LLVM installation lacked the AMDGPU target required by Triton's
multi-backend native extension. LLVM was reconfigured in place with
`Native;NVPTX;AMDGPU`; the resulting incremental build completed 387/387
steps. The following Triton incremental build completed its 8/8 targets.

This was a build-configuration dependency, not an SM103 code-generation or
runtime failure. No compiler source change was used to bypass it.

## Tools

| Tool | Recorded version |
|---|---|
| CMake | 3.31.10 |
| Ninja | 1.13.0 |
| Python | 3.14.6 |
| lit | 18.1.8 |
| psutil | 7.2.2 |
| `ptxas` | CUDA 13.3, V13.3.73 |
| `cuobjdump` | CUDA 13.3, V13.3.73 |
| `nvdisasm` | CUDA 13.3, V13.3.73 |

The focused Triton suite used venv lit 18 with `triton-opt` from the Triton
source build and `FileCheck`/`not` from the locked LLVM source build. The
complete runner record is [raw/focused-lit.log](raw/focused-lit.log).

## Validation results

| Check | Result |
|---|---|
| Focused compiler suite | PASS: 21/21 lit files, representing 29 `RUN:` lines |
| `tma_matmul`, WS off | PASS: compile, full pass trace, disassembly, validation |
| `tma_matmul`, WS on | PASS: compile, full pass trace, disassembly, validation |
| `ws_skeleton`, WS off | PASS: compile, full pass trace, disassembly, validation |
| `ws_skeleton`, WS on | PASS: compile, full pass trace, disassembly, validation |
| Artifact safety policy | PASS: cubin loaded `false`, cubin launched `false` |

Each compile record reports `compiled-not-launched` and `cuda:103:32`; each PTX
file declares `.target sm_103a`. Trace files are SHA-256-bound by their compile
records, validation reports contain no errors, and disassembly records report
`disassembled-not-executed`.

## Claim boundary

These results establish that the frozen compiler stack builds, that the
selected transformations pass their focused regression tests, and that the
four study variants produce the expected compile-time IR/PTX/SASS structures.
They do not establish SM103 silicon correctness, deadlock freedom, occupancy,
latency, throughput, or a warp-specialization speedup. No cubin was loaded or
launched.

The manifest's `tcgen05` contract is specifically MMA issue plus completion
commit. Post-run comparison with the PTX ISA canonical cross-thread
MMA-to-TMEM-load sequence found no `tcgen05.fence::after_thread_sync` in the
frozen source, focused tests, or exact SM103 PTX. Therefore the PASS table above
must not be read as qualification of the complete TCGen05 inter-thread ordering
contract; this is a recorded implementation/coverage gap pending authoritative
resolution and same-target runtime qualification.
