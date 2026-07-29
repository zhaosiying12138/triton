# Warp specialization source-query record

## Frozen scope

- Repository revision: `bf64a5db1bc8aab0fd4f0076e60f6c367852e47d`
- Evidence target: SM103, compile-only
- Evidence products: TTIR/TTGIR/LLVM IR/PTX structure, pass invariants, compile-time diagnostics, and lit/FileCheck coverage
- Explicitly out of scope: hardware execution, benchmark values, profiler counters, throughput or latency claims, and conclusions that require a physical GPU
- Access date for external sources: 2026-07-30

The ledger accepts an external claim only when it can be joined to at least one symbol or test present at the frozen revision. A merged PR is not sufficient by itself: its described mechanism must still be recognizable in current code. A roadmap item is not treated as implemented unless the frozen tree contains the corresponding pass, operation, or test.

## Inclusion gate

Every ledger entry must satisfy all of the following:

1. The source is one of:
   - `triton-lang/triton` GitHub issue or pull request;
   - NVIDIA PTX ISA, CUDA Programming Guide, or Blackwell documentation;
   - official Triton documentation;
   - official PyTorch/Triton engineering article;
   - the Tawa paper.
2. The claim is relevant to the SM103 compile path.
3. The frozen tree has a concrete source symbol and compile-time test that supports or exercises the claim.
4. The wording distinguishes source fact from the article author's scoped inference.
5. The wording makes no runtime or performance assertion.

`exclusion_reason` remains `null` for every admitted record. Rejected candidates are documented here rather than mixed into the admitted ledger.

## Local source queries

### Revision and working-tree identity

```bash
git rev-parse HEAD
git status --short --branch
git show -s --format='%H%n%ad%n%s' --date=iso-strict HEAD
```

### AutomaticWS pipeline and target selection

```bash
rg -n 'add_warp_specialize|add_optimize_partition_warps|add_allocate_warp_groups|add_warp_specialize_to_llvm' \
  third_party/nvidia/backend/compiler.py

rg -n 'AutomaticWarpSpecialization|createTritonGPUPartitionScheduling|createNVWSInsertAref|createNVWSInsertTmemAref|createNVWSLowerAref|createTritonGPUPartitionLoops|createNVWSLowerWarpGroup' \
  lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp

rg -n 'sm_arch_from_capability|make_ptx|make_cubin|gpu-name' \
  third_party/nvidia/backend/compiler.py
```

Questions answered:

- What exact pass order is active at the frozen revision?
- Which internal partition attributes are verified and later cleared?
- Where are logical partitions converted to physical warp groups and PTX-facing control flow?
- Does the compile-only harness truly request capability 103 at its final target stage?

### PartitionScheduling

```bash
rg -n 'buildGraph|initialDataValues|initialPartitionAssignment|mergePartitions|propagatePartitions|duplicateCheapOps|hasEligibleMemoryOps|cloneMultiPartitionDataOps' \
  lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp

rg -n 'getNodeFlags|computeCost|DescriptorLoadLikeOpInterface|DescriptorStoreLikeOpInterface|MMAv5OpInterface|TMEMLoadOp|TMEMStoreOp' \
  lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionSchedulingUtility.cpp

rg -n 'attention_forward|optimize_broadcast|clone_multi_partition_repeated_users|mma_no_memory_ops|scaled_mma|persistent' \
  test/TritonGPU/partition-scheduling.mlir
```

Questions answered:

- Which operations are data roots and which flags drive initial partitions?
- Which ordered heuristics merge partitions, and which constraints prohibit merges?
- What is the current descriptor/TMA eligibility gate?
- Which current tests cover attention, scaled MMA, persistent loops, no-root fallback, and rewrite robustness?

### Partition attributes and PartitionLoops

```bash
rg -n 'verifyPartitionAttrs|verifyPartitionedLoop|setPartition|setPartitionOutputs|setWarpSpecializeTag' \
  lib/Dialect/TritonGPU/Transforms/WarpSpecialization/Partition.cpp

rg -n 'classifyLoopVars|cloneForOp|cloneIfOp|cloneReduceOp|cloneOpsInBlock|partitionLoop' \
  lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionLoops.cpp

rg -n 'multiple_partitions|split_block_arguments|partition_outputs|tensor_captures_over_smem|if_stmt_split|still_has_ssa_deps' \
  test/TritonGPU/partition-loops.mlir

rg -n 'partition ids|verified only when consumed|CLEAN' \
  test/TritonGPU/partition-verifier-locality.mlir \
  test/TritonGPU/automatic-warp-specialization.mlir
```

Questions answered:

- Which annotations are analysis-only state?
- How are loop arguments, induction variables, captures, and results classified?
- How are `scf.for`, `scf.if`, and reductions reconstructed per partition?
- Which malformed state is rejected before materialization?

### ARef and TMEM ARef

```bash
rg -n 'ArefCreateOp|ArefPutEnterOp|ArefPutExitOp|ArefGetEnterOp|ArefGetExitOp|ArefBufferOp' \
  third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td

rg -n 'getProducedValues|createArefPut|createArefGet|insertArefs|getConsumerAsyncOpKinds' \
  third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertAref.cpp

rg -n 'TmemAccessDag|TMEMAref|hasProducerConsumerPartitioning|workaroundForLoopScheduler|runOnFunction' \
  third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertTmemAref.cpp

rg -n 'ArefValue|createBarriers|lowerTMALoad|rewritePutEnterOp|rewritePutExitOp|rewriteGetEnterOp|rewriteGetExitOp|multiBufferAref|combineArefs' \
  third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp

rg -n 'warp_specialize_tma_matmul|two_consumers|different_yield_partition|aref_result_outside_scheduled_loop|nested_loop|test_tmem_no_ws' \
  test/NVWS/insert_aref.mlir \
  test/NVWS/aref-tmem-insertion.mlir \
  test/NVWS/lower_aref.mlir
```

Questions answered:

- Which SSA values become channels?
- How are producer/consumer multiplicity, buffer index, stage, and phase represented?
- When is shared memory sufficient, and when does TMEM ownership require a TMEM ARef?
- What are the create/use/wait/arrive/invalidate lifetime boundaries?
- Which cases intentionally decline to create TMEM ARefs?

### TCGen05 and scaled MMA

```bash
rg -n 'TCGen5MMAOp|TCGen5MMAScaledOp|MMAv5OpInterface|createGen5MMA|createScaledGen5MMA|createMMACommit|convertScaledDot' \
  lib/Dialect/TritonGPU/Transforms/WarpSpecialization \
  lib/Dialect/TritonNvidiaGPU/Transforms/MMALowering.cpp \
  third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp

rg -n 'TensorMemoryLoadOpConversion|TensorMemoryStoreOpConversion|TensorMemoryAllocOpConversion|createTcgen05Cp' \
  third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp

rg -n 'tc_gen5_mma|tc_gen5_commit|tensor_memory_ld|scaled|cuda:103' \
  test/Conversion/tritongpu_to_llvm_blackwell.mlir \
  test/Conversion/lower_tensor_memory_to_llvm.mlir \
  test/TritonNvidiaGPU/fuse_tmem_load_reduce.mlir
```

Questions answered:

- Which TCGen05 operation or token is considered a partition data root?
- Where are scale descriptors and scale TMEM operands lowered?
- Which compile-time checks prove emission of `tcgen05.mma`, `tcgen05.commit`, `tcgen05.ld/st/cp`, and their scaled forms for capability 103?
- Which lower-level capability-103 tests exist independently of the mostly capability-100 AutomaticWS transform tests?

### TMA, barriers, fences, and hazards

```bash
rg -n 'TMALoadLowering|TMAGatherLowering|TMAStoreLowering|TMAReduceLowering|TMAScatterLowering' \
  lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp

rg -n 'ProxyFenceAnalysis|insertFence|TMemBarrierAnalysis|insertBarrier' \
  lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp \
  lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp

rg -n 'mma_inside_warp_specialize|matmul_like_fence_mma_v5' \
  test/TritonGPU/fence-inserstion.mlir

rg -n 'cluster_barrier|tc_gen5_commit|tma_completion|barrier_reinit|wait_barrier_without_init|arrive_barrier_without_init' \
  test/TritonGPU/consan.mlir \
  test/TritonNvidiaGPU/membar-cluster.mlir \
  test/TritonNvidiaGPU/cluster-barrier-mbar-allocator.mlir
```

Questions answered:

- Which synchronization expresses event completion and which expresses memory-proxy visibility?
- Where are TMEM read/write hazards repaired?
- Which barriers must execute outside a subset-only partition?
- How are invalid initialization, reinitialization, and incomplete completion detected by compile-time instrumentation?

### Register budgets and physical warp groups

```bash
rg -n 'optimizePartitionNumWarps|getTensorNumI32Regs|tmem_min_4_warps' \
  lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp \
  test/TritonGPU/optimize-partition-warps.mlir

rg -n 'padToMaxWarpGroups|maxnreg|requestedRegisters|actualRegisters|warpGroupStartIds' \
  lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp

rg -n 'createRegRealloc|SetMaxRegister|WorkerPartitionStart|DefaultPartitionStart|lowRegs|defRegs' \
  third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp

rg -n 'setmaxnreg|steal_from_default|dynamic_register_reallocation' \
  test/Conversion/allocate_warp_groups.mlir \
  test/Conversion/warp_specialize_to_llvm.mlir
```

Questions answered:

- How are logical partition warp counts estimated and changed?
- Why are partitions padded to full physical warpgroups?
- How are requested register estimates converted into actual per-region budgets?
- At which control-flow points are `setmaxnreg.inc/dec` emitted?

### Persistent attention and grouped GEMM

```bash
rg -n 'attention_forward|attention_persistent_inner_loop_kernel|grouped_matmul_tma_kernel|matmul_nested_persistent_ws_kernel' \
  test/TritonGPU/automatic-warp-specialization.mlir \
  test/TritonGPU/partition-scheduling.mlir

rg -n 'canProveExecuteOnce|hoistTmemAlloc|underWSLoop' \
  third_party/nvidia/lib/Dialect/NVWS/Transforms/HoistTmemStore.cpp

rg -n 'visitBackwardSlice|updateOutputWithDefaultPartition|assignStagePhase' \
  third_party/nvidia/lib/Dialect/NVWS/Transforms/AssignStagePhase.cpp
```

Questions answered:

- How does nested-loop partition propagation differ from single-loop matmul?
- Under what proof can TMEM initialization be hoisted?
- How are block-argument producers traced through yields in attention?
- Which grouped-GEMM-shaped compile-time case reaches the full AutomaticWS pipeline?

### Verifier and sanitizer

```bash
rg -n 'VerifyWarpSpecializationPartitions|addPassWithPartitionVerifier|clearInternalWarpSpecializationAttrs' \
  lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp

rg -n 'ConcurrencySanitizerImpl|NVIDIAConSanHooks|PrepareConSanCaptures' \
  lib/Dialect/TritonInstrument/Transforms \
  lib/Dialect/TritonNvidiaGPU/Transforms/ConSanNVIDIA.cpp

rg -n 'cluster_barrier_partition_scopes|proxy_fence_state_transitions|tma_completion_tracks_contained_proxy_frontier|tcgen5_mma|barrier_reinit_requires_invalidate' \
  test/TritonGPU/consan.mlir
```

Questions answered:

- Which invariants are verified after each AutomaticWS subpass?
- Which internal attributes are guaranteed not to escape the pass?
- What synchronization and memory effects can ConSan model at compile time?

## GitHub issue and PR queries

The following queries were applied to `triton-lang/triton`, then candidates were checked against local ancestry and current symbols:

```text
repo:triton-lang/triton is:issue "warp specialization" Blackwell
repo:triton-lang/triton is:pr "Automatic warp specialization"
repo:triton-lang/triton is:pr "PartitionScheduling"
repo:triton-lang/triton is:pr "PartitionLoops"
repo:triton-lang/triton is:pr ARef NVWS
repo:triton-lang/triton is:pr "InsertTmemAref"
repo:triton-lang/triton is:pr TCGen5 scaled partition
repo:triton-lang/triton is:pr TMA "warp specialization"
repo:triton-lang/triton is:pr setmaxnreg
repo:triton-lang/triton is:pr "warp group" registers
repo:triton-lang/triton is:pr proxy fence "warp specialization"
repo:triton-lang/triton is:pr mbarrier "warp specialization"
repo:triton-lang/triton is:pr attention "warp specialization"
repo:triton-lang/triton is:pr grouped GEMM "warp specialization"
repo:triton-lang/triton is:pr ConSan "WarpSpecialization"
repo:triton-lang/triton is:pr verifier partition attrs WS
```

Local ancestry check:

```bash
git log --all --date=short --pretty='%h %ad %s' \
  --grep='warp special\|AutomaticWS\|ARef\|partition schedul\|tmem.*aref\|setmaxnreg\|warp group\|TCGen5\|scaled MMA\|cluster barrier\|ConSan' -i

git merge-base --is-ancestor <candidate-commit> bf64a5db
```

PR review text is used for design motivation or maintainer judgment. The frozen implementation and tests remain authoritative if an old review describes an earlier API or pass order.

## NVIDIA official-source queries

```text
site:docs.nvidia.com/cuda/parallel-thread-execution tensor memory tcgen05 issue granularity
site:docs.nvidia.com/cuda/parallel-thread-execution tcgen05 mma commit fence wait
site:docs.nvidia.com/cuda/parallel-thread-execution mbarrier phase tx-count expect-tx
site:docs.nvidia.com/cuda/parallel-thread-execution proxy fence async generic
site:docs.nvidia.com/cuda/parallel-thread-execution setmaxnreg warpgroup
site:docs.nvidia.com/cuda/cuda-programming-guide TMA elected thread mbarrier
site:docs.nvidia.com/cuda/cuda-programming-guide producer consumer warp specialization
site:docs.nvidia.com/cuda/cuda-programming-guide compute_100f compute_103f
site:docs.nvidia.com/cuda/blackwell-tuning-guide register file thread block clusters
```

Accepted NVIDIA pages:

- PTX ISA 9.3 Tensor Memory and TCGen05 instruction sections
- PTX ISA 9.3 `mbarrier`, named barrier, proxy-fence, and `setmaxnreg` sections
- CUDA Programming Guide TMA and producer-consumer sections
- CUDA Programming Guide compute-capability and feature-set target table
- Blackwell Tuning Guide resource and cluster sections

ISA claims are used only to explain why the corresponding Triton lowering or verifier exists. The article will not infer successful execution from legal-looking PTX alone.

## Triton, PyTorch, and Tawa queries

```text
site:triton-lang.org/main/python-api/generated/triton.language.range.html warp_specialize
site:triton-lang.org/main/dialects/TritonGPUOps.html ttg.warp_specialize
site:pytorch.org/blog "Warp Specialization in Triton: Design and Roadmap"
"Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References" PDF
```

Reconciliation rules:

- `triton.language.range` defines the public promise and is intentionally more conservative than all cases found in internal tests.
- TritonGPU operation documentation defines the current TTGIR contract.
- The January 2026 PyTorch article is treated as a dated architecture/roadmap snapshot; frozen source decides which listed plans have landed.
- Tawa supplies the conceptual origin of ARef and task-aware partitioning. Its evaluated hardware and reported results are not used here.

## Coverage matrix

| Topic | Primary frozen symbols | Compile-time witnesses |
|---|---|---|
| AutomaticWS orchestration | `AutomaticWarpSpecialization::runOnOperation` | `automatic-warp-specialization.mlir` |
| Partition scheduling | `buildGraph`, `mergePartitions`, `propagatePartitions` | `partition-scheduling.mlir` |
| Partition materialization | `partitionLoop`, `cloneForOp`, `cloneIfOp` | `partition-loops.mlir` |
| ARef | `insertArefs`, `createBarriers`, `multiBufferAref` | `insert_aref.mlir`, `lower_aref.mlir` |
| TMEM ARef | `TmemAccessDag`, `TMEMAref`, `runOnFunction` | `aref-tmem-insertion.mlir` |
| TCGen05/scaled | `createGen5MMA`, `createScaledGen5MMA`, `createMMACommit` | `tritongpu_to_llvm_blackwell.mlir` |
| TMA | `TMALoadLowering`, `lowerTMALoad` | `tma_to_llvm.mlir`, `lower_aref.mlir` |
| mbarrier/fence | `createBarriers`, `ProxyFenceAnalysis`, `TMemBarrierAnalysis` | `consan.mlir`, `fence-inserstion.mlir` |
| warp groups/registers | `OptimizePartitionWarps`, `AllocateWarpGroups`, `createRegRealloc` | `optimize-partition-warps.mlir`, `allocate_warp_groups.mlir`, `warp_specialize_to_llvm.mlir` |
| persistent attention | `visitBackwardSlice`, `hoistTmemAlloc` | persistent-attention cases in `partition-scheduling.mlir` |
| grouped GEMM | graph propagation plus nested `PartitionLoops` | `grouped_matmul_tma_kernel` in `automatic-warp-specialization.mlir` |
| verifier/sanitizer | `VerifyWarpSpecializationPartitions`, `ConcurrencySanitizerImpl` | `partition-verifier-locality.mlir`, `consan.mlir` |

## Known evidence gaps at the frozen revision

1. Most end-to-end AutomaticWS lit inputs identify the target as `cuda:100`. Capability-103 lowering is covered separately by TMEM/TCGen05 tests, but there is no single checked-in lit case that starts with an AutomaticWS loop, carries target `cuda:103` through every pass, and FileChecks the final PTX.
2. `sm_arch_from_capability` still contains `TODO: Handle non-"a" sms`. A compile-only study must record the exact emitted `.target` and ptxas target rather than infer family-target behavior from CUDA documentation.
3. The repository contains strong lit coverage for individual transformations, but no frozen artifact currently captures one complete case at every boundary: pre-scheduling TTGIR, partition attributes, ARef, partitioned TTGIR, LLVM IR, and PTX.
4. ConSan tests validate instrumented IR transformations and modeled synchronization semantics. Without execution, they do not prove absence of hardware races or deadlocks.
5. Persistent attention and grouped GEMM have compile-time regression cases, but the public `tl.range` documentation still advertises only simple matmul loops. The article must distinguish implemented regression coverage from the documented user-facing guarantee.
6. PartitionScheduling is heuristic. The frozen tree verifies legality and many structural outcomes, but it has no formal optimality proof or general profitability model.
7. The compile-only boundary permits claims such as “the intended partition/channel/instruction structure was generated.” It does not permit claims of numerical correctness, progress on hardware, occupancy, latency hiding, or speedup.

## Article claim labels

Use these labels consistently:

- **Source fact**: directly stated by an admitted external source and visible in frozen code/tests.
- **Frozen-source observation**: derived by reading a symbol or FileCheck at `bf64a5db`.
- **Scoped inference**: a mechanism-level conclusion joining an external hardware contract to a frozen lowering; it must be worded as an inference.
- **Not established compile-only**: any runtime correctness, liveness, or performance conclusion.

