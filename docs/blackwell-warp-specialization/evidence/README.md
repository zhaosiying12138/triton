# Compile-only evidence

This directory is the versionable evidence boundary for the Blackwell
warp-specialization study. The recorded run used Triton
`bf64a5db1bc8aab0fd4f0076e60f6c367852e47d`, LLVM
`850a2b1b975c061ae0fc982ba68064d305485cb2`, and the synthetic target
`cuda:103:32`.

The checked-in records are:

- [BUILD_PROVENANCE.md](BUILD_PROVENANCE.md): source locks, source-build
  configuration, the resolved LLVM target-set issue, tool versions, and the
  validation boundary;
- [raw/focused-lit.log](raw/focused-lit.log): the complete focused lit output,
  with 31/31 files passing and 43 `RUN:` lines represented; the selection now
  includes every file in `test/NVWS`;
- [sm103-compile-only/evidence.md](sm103-compile-only/evidence.md): the reviewed
  four-variant result table, with matching JSON and standalone HTML beside it.

The four compile records (`tma_matmul` and `ws_skeleton`, each with WS on and
off), their hash-bound full pass traces, cubins, disassemblies, validation
records, and review render exist under the ignored `../lab/build/` audit tree.
All four compiled and validated successfully as `cuda:103:32`; disassembly
also succeeded. The harness recorded every cubin as not loaded and not
launched.

No SM103 runtime result or performance measurement is claimed. Lit wall times
in the raw runner output are test-runner diagnostics, not kernel-performance
data.

## Known qualification gap found after publication

The `tcgen05` manifest family asserts MMA issue plus completion commit; it does
not assert the complete PTX ISA cross-thread ordering pattern. A subsequent ISA
audit found that both `tma_matmul` variants contain `tcgen05.commit`, mbarrier
wait, CTA synchronization, `tcgen05.ld`, and `tcgen05.wait::ld`, but omit the
canonical `tcgen05.fence::after_thread_sync` between completion synchronization
and the load. The frozen Triton dialect/lowering/tests also contain no emission
or check for that instruction.

Consequently, the four PASS records remain valid for the exact structural
contracts declared in `manifest.json`, while TCGen05 cross-role ordering and
runtime correctness remain explicitly unqualified. `ptxas` acceptance does not
close that semantic gap.

## Reproduce or republish

The reviewed render has been published into `sm103-compile-only/`. To reproduce
it from the ignored raw audit tree, validate and render before invoking the
explicit publication step:

```bash
cd ../lab
python3 ws_study.py validate --require-artifacts
python3 ws_study.py render
python3 ws_study.py render --publish
```

Publication refuses to replace different existing evidence unless the reviewer
passes `--force`. Raw compiler outputs and review staging files remain ignored
under `../lab/build/`; only deliberately selected evidence belongs here.
