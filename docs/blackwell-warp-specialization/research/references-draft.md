# 参考资料与证据索引

本章是全文唯一集中列出外部链接、PR 和 issue 编号的位置。源码结论固定到 Triton 提交 `bf64a5db1bc8aab0fd4f0076e60f6c367852e47d`；每组先给出可复查的冻结源码与测试，再列一手外部来源。外部来源用于回答设计动机、硬件契约或维护者判断，实际 pass 顺序、symbol 和已覆盖 case 仍以冻结树为准。

## 硬件语义：SM103、TMA、TCGen05 与 TMEM

冻结源码与测试：

- `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp::getMMAVersionSafe`、`include/triton/Dialect/TritonNvidiaGPU/IR/TargetFeatures.h::TargetFeatures` 与 `test/Conversion/tritongpu_to_llvm_sm120.mlir`：本文用它们回答为何 consumer Blackwell SM120 的 MMA v2/cluster feature path 不能替代 SM103 的 TCGen05/TMEM WS 实验。
- `third_party/nvidia/backend/compiler.py::{sm_arch_from_capability,CUDABackend.make_ttgir,CUDABackend.make_ptx,CUDABackend.make_cubin}`：本文用它回答 SM103 目标如何进入 Blackwell pass pipeline，以及最终 `.target` 和 ptxas target 在哪里确定。
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp::{createGen5MMA,createScaledGen5MMA,createMMACommit}`：本文用它回答 `ttng.tc_gen5_mma{_scaled}` 如何生成 TCGen05 MMA、descriptor 和 completion commit。
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp::{TensorMemoryLoadOpConversion,TensorMemoryStoreOpConversion,TensorMemoryAllocOpConversion,createTcgen05Cp}`：本文用它回答 TMEM 分配、寄存器读写和 SMEM→TMEM copy 如何物化。
- `lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp::{TMALoadLowering,TMAGatherLowering,TMAStoreLowering}`：本文用它回答 tensor descriptor 操作如何变成带完成通知的 TMA 操作。
- `test/Conversion/tritongpu_to_llvm_blackwell.mlir::{tc_gen5_mma,tc_gen5_commit,tensor_memory_ld,tc_gen5_mma_scaled_fp8_tmem_lhs}`：本文用它回答编译结果中应出现哪些 TCGen05/TMEM 指令形态。
- `test/TritonNvidiaGPU/fuse_tmem_load_reduce.mlir`、`test/Conversion/lower_tensor_memory_to_llvm.mlir`：本文用它回答 capability 103 下 TMEM reduction、allocation、deallocation 与 permit relinquish 的 lowering 是否有直接回归覆盖。
- `test/Conversion/tma_to_llvm.mlir`：本文用它回答 TMA gather/scatter 的独立 lowering 契约。

外部来源：

- [CUDA Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)：本文用它回答 `compute_100f`、`compute_103f` 与 architecture-specific target 的兼容边界，避免把 major-version 相同误写成 feature set 完全相同。
- [PTX ISA：Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory)：本文用它回答 TMEM 的存储角色、访问粒度及其与第五代 Tensor Core 的关系。
- [PTX ISA：Fifth-Generation TensorCore Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions)：本文用它回答 TCGen05 MMA、commit、fence、wait、alloc/dealloc 和 issue granularity 的硬件契约。
- [CUDA Programming Guide：Tensor Memory Accelerator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-the-tensor-memory-accelerator-tma)：本文用它回答 TMA 为什么可以由 elected thread 发起，以及 expected bytes 与 mbarrier completion 如何配合。
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)：本文用它回答 SM103 compile plan 所受寄存器文件、每线程寄存器上限、shared memory 与 cluster 资源边界。

## 执行结构：从前端意图到 default/worker partitions

冻结源码与测试：

- `python/triton/language/core.py::range`：本文用它回答 `warp_specialize=True` 在用户层只表达“请求编译器尝试分区”，而不是保证最终一定 materialize。
- `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td::{WarpSpecializeOp,WarpSpecializePartitionsOp,WarpYieldOp,WarpReturnOp}`：本文用它回答 default region、isolated worker regions、captures、warp counts 和返回值的 TTGIR ABI。
- `lib/Dialect/TritonGPU/IR/Ops.cpp::{WarpSpecializeOp::verify,WarpSpecializePartitionsOp::verify}`：本文用它回答 partition 数量、warp 数、captures、嵌套限制和 region 结构如何验证。
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerWarpGroup.cpp::LowerWarpGroup`：本文用它回答临时 `nvws.warp_group` 如何转换成正式 `ttg.warp_specialize`。
- `lib/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.cpp::lowerWarpSpecializeCommon`、`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp::lowerWarpSpecialize`：本文用它回答 warp-id dispatch、worker switch loop、join barrier 和 partition state 如何生成。
- `test/TritonGPU/automatic-warp-specialization.mlir`：本文用它回答 AutomaticWS 后是否真的出现 `ttg.warp_specialize`，以及内部 partition attributes 是否被清理。
- `test/Conversion/warp_specialize_to_llvm.mlir::{rewrite_barriers,warpid_warp_specialize}`：本文用它回答 TTGIR partition 怎样变成 LLVM 控制流和 barrier protocol。

外部来源：

- [TritonGPUOps：`ttg.warp_specialize`](https://triton-lang.org/main/dialects/TritonGPUOps.html#ttg-warp-specialize)：本文用它回答当前公开 IR 中 `partitionNumWarps`、`warpGroupStartIds`、`requestedRegisters`、`actualRegisters` 与 explicit captures 的定义。
- [Triton `tl.range`](https://triton-lang.org/main/python-api/generated/triton.language.range.html)：本文用它回答用户可见 API 的当前支持承诺及“可能增加 kernel 总 warp 数”的语义。
- [triton-lang/triton#4308](https://github.com/triton-lang/triton/issues/4308)：本文用它回答为什么编译器需要把 TMA、tensor-core 与 attention-side computation 变成可重叠的不同执行角色。
- [triton-lang/triton#5917](https://github.com/triton-lang/triton/pull/5917)：本文用它回答 `ttg.warp_specialize` 为什么采用 default region 加 worker partitions 的 IR 结构。
- [triton-lang/triton#5968](https://github.com/triton-lang/triton/pull/5968)：本文用它回答 partition IR 需要哪些 warp dispatch、barrier 和 worker-loop lowering。
- [triton-lang/triton#6217](https://github.com/triton-lang/triton/pull/6217)：本文用它回答 descriptor-load→MMAv5 simple matmul 为什么是 AutomaticWS 的基准正向 case。

## PartitionScheduling 与 PartitionLoops

冻结源码与测试：

- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp::{buildGraph,initialDataValues,initialPartitionAssignment,mergePartitions,propagatePartitions,duplicateCheapOps,hasEligibleMemoryOps}`：本文用它回答数据流图如何建立、data roots 如何选取、heuristics 如何合并 partition，以及何时判定没有可用 memory root。
- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionSchedulingUtility.cpp::{getNodeFlags,computeCost}`：本文用它回答 descriptor load/store、MMAv5、TMEM、SFU、view 分别如何分类和估算成本。
- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/Partition.cpp::{PartitionSet::fromLoop,verifyPartitionedLoop,setPartition,setPartitionOutputs}`：本文用它回答 partition attributes 的一致性、不跨越非法 SSA edge 的约束和 output metadata。
- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionLoops.cpp::{classifyLoopVars,cloneForOp,cloneIfOp,cloneReduceOp,partitionLoop}`：本文用它回答 loop args、results、captures 与结构化控制流怎样被逐 partition 重建。
- `test/TritonGPU/partition-scheduling.mlir::{attention_forward,optimize_broadcast,mma_no_memory_ops,scaled_mma_with_loads,clone_multi_partition_repeated_users}`：本文用它回答 scheduler 的正向、fallback、局部通信优化与 rewrite 稳健性 case。
- `test/TritonGPU/partition-loops.mlir::{multiple_partitions,split_block_arguments,partition_outputs,tensor_captures_over_smem,if_stmt_split,still_has_ssa_deps}`：本文用它回答 PartitionLoops 对各类 SSA 与 control-flow 形态的物化边界。
- `test/TritonGPU/partition-verifier-locality.mlir`：本文用它回答 malformed 或跨域 partition metadata 在何处被拒绝。

外部来源：

- [triton-lang/triton#6175](https://github.com/triton-lang/triton/pull/6175)：本文用它回答为什么 AutomaticWS 必须被建模为带 verifier 的 dataflow/SSA transformation。
- [triton-lang/triton#6186](https://github.com/triton-lang/triton/pull/6186)：本文用它回答为什么依赖重写必须先于 default/worker region materialization。
- [triton-lang/triton#7312](https://github.com/triton-lang/triton/pull/7312)：本文用它回答当前 graph-based scheduler 的 heuristic merge 策略，以及为什么它被限定为 loop-local optimization。
- [triton-lang/triton#7415](https://github.com/triton-lang/triton/pull/7415)：本文用它回答 `scf.for`、`scf.if` 与其他结构化控制流为何需要递归重建而非平面 clone。
- [triton-lang/triton#9716](https://github.com/triton-lang/triton/pull/9716)：本文用它回答为什么至少需要一个 eligible descriptor memory operation，以及 pointer-only dataflow 为什么不能被当作 canonical positive case。
- [triton-lang/triton#11067](https://github.com/triton-lang/triton/pull/11067)：本文用它回答 multi-partition data-op cloning 为什么必须在修改 use-list 时使用稳定遍历。

## ARef 与 TMEM ARef

冻结源码与测试：

- `third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td::{ArefCreateOp,ArefPutEnterOp,ArefPutExitOp,ArefGetEnterOp,ArefGetExitOp,ArefBufferOp}`：本文用它回答 ARef 在 IR 中怎样表达 payload、producer/consumer 临界区与逻辑 channel 生命周期。
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertAref.cpp::{getProducedValues,createArefPut,createArefGet,insertArefs}`：本文用它回答哪些跨 partition SSA values 会变成 ARef，以及多 producer/consumer、yield 和 last-use 如何处理。
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertTmemAref.cpp::{TmemAccessDag,TMEMAref,hasProducerConsumerPartitioning,runOnFunction}`：本文用它回答 MMAv5→TMEM consumer 的 ownership/hazard graph 如何与普通 SMEM ARef 区分。
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp::{ArefValue,createBarriers,rewritePutEnterOp,rewritePutExitOp,rewriteGetEnterOp,rewriteGetExitOp,multiBufferAref,combineArefs}`：本文用它回答 ARef 如何降低为 buffer、empty/full barriers、stage index、phase、wait、arrive 与 invalidate。
- `test/NVWS/ops.mlir`：本文用它回答 ARef 与 warp-group ops 的基本 verifier contract。
- `test/NVWS/insert_aref.mlir::{two_consumers,different_yield_partition,conditional_consumer,aref_result_outside_scheduled_loop}`：本文用它回答复杂 SSA producer/consumer 关系何时可合法建立 channel。
- `test/NVWS/aref-tmem-insertion.mlir::{store_mma_load,nested_loop_yes_double_buffer,nested_loop_no_double_buffer,test_tmem_no_ws}`：本文用它回答 TMEM ARef 的双缓冲、nested-loop 与 no-materialization 分支。
- `test/NVWS/lower_aref.mlir::{warp_specialize_tma_matmul,load_used_as_reg_and_smem,attention_forward}`：本文用它回答 channel lowering 对 TMA、register+SMEM 双重使用和 attention 的具体 IR 效果。

外部来源：

- [Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References](https://www.csl.cornell.edu/~zhiruz/pdfs/tawa-cgo2026.pdf)：本文用它回答 ARef 的 one-slot channel、empty/full credit、put/get/consumed 和循环多缓冲的抽象来源。
- [triton-lang/triton#6288](https://github.com/triton-lang/triton/pull/6288)：本文用它回答为什么 Triton 引入 NVWS dialect 和 ARef，而不是在 partition scheduler 中直接拼接 raw barriers。
- [triton-lang/triton#7479](https://github.com/triton-lang/triton/pull/7479)：本文用它回答 buffer index 为什么同时决定 stage slot 与 mbarrier phase。
- [triton-lang/triton#7645](https://github.com/triton-lang/triton/pull/7645)：本文用它回答 cross-partition SSA 如何转换成 ARef put/get，以及 barrier storage 重用前为什么需要失效旧生命周期。
- [triton-lang/triton#8262](https://github.com/triton-lang/triton/pull/8262)：本文用它回答 ARef insertion/lowering 在当前 AutomaticWS pipeline 中所处的位置。
- [triton-lang/triton#9007](https://github.com/triton-lang/triton/pull/9007)：本文用它回答 TMEM async state 为什么必须按 partition 保存，以及“请求 WS”为什么不等于“已经生成 partition”。
- [triton-lang/triton#9114](https://github.com/triton-lang/triton/pull/9114)：本文用它回答 loop 外 result 仍可创建 ARef channel，但只让 scheduled-loop 内 results 参与 enter/exit stage-cluster 推导。

## Software pipeline 与 stage/cluster schedule

冻结源码与测试：

- `third_party/nvidia/backend/compiler.py::CUDABackend.make_ttgir`：本文用它回答 SM103 pipeline 中 AssignLatencies、ScheduleLoops、AutomaticWS、Pipeline 与 OptimizePartitionWarps 的实际次序。
- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp::AutomaticWarpSpecialization::runOnOperation`：本文用它回答 AutomaticWS 内部 ARef、PartitionLoops、LowerWarpGroup 与二次 ScheduleLoops 的次序。
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/AssignStagePhase.cpp::{assignStagePhase,visitBackwardSlice,updateOutputWithDefaultPartition}`：本文用它回答 ARef operation 的 stage/cluster 如何从真实 producer 与 consumer schedule 推导。
- `lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp`、`ScheduleLoops.cpp`、`LowerLoops.cpp`：本文用它回答 latency、coarse schedule、self-latency 与 wait insertion 如何协作。
- `test/TritonGPU/automatic-warp-specialization.mlir` 的 `BASE`、`PIPELINE`、`OPT` checks：本文用它回答同一输入在 partition、pipeline 和 warp-count optimization 后分别达到什么 IR 状态。
- `test/NVWS/assign_stage_phase.mlir::{assign_stage_buffer,attention_forward,for_loop_control_operand_ppg}`：本文用它回答 stage/phase 对 buffer、attention block arg 与 loop-control operand 的传播。
- `test/TritonGPU/pipeline-assign-latencies.mlir`、`test/TritonGPU/pipeline-schedule-loop.mlir`：本文用它回答 mixed operand 与 self-latency 的调度依据。

外部来源：

- [triton-lang/triton#6887](https://github.com/triton-lang/triton/pull/6887)：本文用它回答 WS 为什么不替代 software pipelining，而是消费其 stage/cluster schedule 并让分区后的 loop 再 pipeline。
- [triton-lang/triton#8883](https://github.com/triton-lang/triton/pull/8883)：本文用它回答 attention block argument 的 producer 为什么必须沿 `scf.yield` 回溯到真实定义。
- [triton-lang/triton#9111](https://github.com/triton-lang/triton/pull/9111)：本文用它回答 mixed TMA/non-TMA operand 为什么需要 self-latency 和显式 wait，而不能假定所有 MMA inputs 同步到达。

## Barrier、completion 与 proxy fence

冻结源码与测试：

- `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp::{createBarriers,insertWaitOp,insertArriveBarrier,lowerTMALoad}`：本文用它回答 ARef 的 empty/full barrier、arrival count、transaction completion 和 phase wait 如何生成。
- `lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp::ProxyFenceAnalysis`：本文用它回答 generic proxy 与 async proxy 间的可见性 hazard 如何独立于 mbarrier completion 修复。
- `lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp::TMemBarrierAnalysis`：本文用它回答 MMA、TMEM load/store 与共享 allocation slices 之间的 RAW/WAR/WAW hazard 如何插 barrier。
- `lib/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.cpp::ClusterBarrierAnalysis`：本文用它回答跨 CTA dependency 为什么需要 all-thread cluster rendezvous，以及 barrier 应放在何种 control-flow 点。
- `lib/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierMbarAllocator.cpp`：本文用它回答 cluster/atomic ordering 与 WS channel 如何共享 mbarrier 资源规划。
- `test/TritonGPU/fence-inserstion.mlir::{matmul_like_fence_mma_v5,mma_inside_warp_specialize}`：本文用它回答 partition 后新增的 async/generic proxy ordering。
- `test/TritonGPU/consan.mlir::{proxy_fence_state_transitions,tma_completion_tracks_contained_proxy_frontier,barrier_reinit_requires_invalidate,wait_barrier_without_init,arrive_barrier_without_init}`：本文用它回答 completion、proxy frontier 和 barrier lifecycle 的静态 instrumentation 证据。
- `test/TritonNvidiaGPU/membar-cluster.mlir`、`test/TritonNvidiaGPU/cluster-barrier-mbar-allocator.mlir`：本文用它回答 cluster barrier placement 与 mbar resource allocation 的 compile-time contract。

外部来源：

- [PTX ISA：mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier)：本文用它回答 pending arrivals、transaction count、phase 与 parity 何时构成完成状态。
- [PTX ISA：barrier instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar-barrier)：本文用它回答 CTA named barrier 的 participant 与 arrive/wait 语义。
- [PTX ISA：Asynchronous Instructions and Memory Consistency](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-instructions)：本文用它回答 completion mechanism 与不同 memory proxy 可见性为何是两类约束。
- [triton-lang/triton#7278](https://github.com/triton-lang/triton/pull/7278)：本文用它回答 loop 被分区后为何需要重新建立 WAR async+generic proxy fence。
- [triton-lang/triton#9456](https://github.com/triton-lang/triton/pull/9456)：本文用它回答 cluster barrier 为什么不能只由一个 worker partition 的线程执行。
- [triton-lang/triton#9591](https://github.com/triton-lang/triton/pull/9591)：本文用它回答错误 mbarrier re-initialization 如何成为 ConSan 可检测的 lifecycle violation。
- [triton-lang/triton#10914](https://github.com/triton-lang/triton/pull/10914)：本文用它回答 atomic acquire/release ordering 为什么也会消耗为 WS 规划的 mbarrier 资源。

## Register redistribution 与 warp-group allocation

冻结源码与测试：

- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp::{getTensorNumI32Regs,optimizePartitionNumWarps,relayoutWarps}`：本文用它回答逻辑 partition warp 数与 register estimate 如何根据 tensor work 调整。
- `lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp::{padToMaxWarpGroups,AllocateWarpGroups::runOnOperation}`：本文用它回答 partition 如何被排列、补齐到物理 4-warp groups，并得到 `warpGroupStartIds`、`actualRegisters` 与 module maxnreg。
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp::{createRegRealloc,lowerWarpSpecialize}`：本文用它回答 default/worker 进入和离开 partition 时怎样发出 `setmaxnreg.inc/dec`。
- `test/TritonGPU/optimize-partition-warps.mlir::{small_tensor_computation,register_use_heuristic,tmem_min_4_warps}`：本文用它回答逻辑 warp shrink、估算规则与 TMEM 的四 warp 下限。
- `test/Conversion/allocate_warp_groups.mlir::{setmaxnreg,steal_from_default}`：本文用它回答 worker/default register budget 和 program-wide padding 的 IR 属性结果。
- `test/Conversion/warp_specialize_to_llvm.mlir::dynamic_register_reallocation`：本文用它回答 register handoff 最终是否降成 NVVM `setmaxregister` 操作。

外部来源：

- [PTX ISA：`setmaxnreg`](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg)：本文用它回答 register pool、`.inc/.dec`、warpgroup 一致执行、数值粒度与同步要求。
- [triton-lang/triton#6323](https://github.com/triton-lang/triton/pull/6323)：本文用它回答为什么 partition warp-count optimization 被拆成独立 pass，以及 TMEM consumer 为什么不能任意缩到单 warp。
- [triton-lang/triton#6407](https://github.com/triton-lang/triton/pull/6407)：本文用它回答 `requestedRegisters` 为什么只是中端 estimate，而非精确寄存器证明。
- [triton-lang/triton#6694](https://github.com/triton-lang/triton/pull/6694)：本文用它回答为什么需要 program-wide warpgroup padding 和 default/worker 的动态 register handoff。
- [triton-lang/triton#8005](https://github.com/triton-lang/triton/pull/8005)：本文用它回答为什么当前 warp-specialized kernel 的 base warp count 必须是四的倍数。

## Persistent matmul、attention、grouped GEMM 与 scaled MMA

冻结源码与测试：

- `third_party/nvidia/lib/Dialect/NVWS/Transforms/HoistTmemStore.cpp::{canProveExecuteOnce,hoistTmemAlloc}`：本文用它回答 nested persistent loop 中 TMEM allocation/initialization 何时可安全 hoist。
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/AssignStagePhase.cpp::visitBackwardSlice`：本文用它回答 attention reduction 的 loop-carried block arg 如何关联到真实 producer schedule。
- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp::{getTMEMAllocs,partition_heuristics,propagatePartitions}`：本文用它回答 MMA、TMEM epilogue、SFU 与 nested control flow 如何形成 case-specific partitions。
- `lib/Dialect/TritonNvidiaGPU/Transforms/MMALowering.cpp::TCGen5MMAScaleSharedToTmemConversion`：本文用它回答 block-scale operands 如何进入 TMEM 并参与 MMAv5 lowering。
- `test/TritonGPU/automatic-warp-specialization.mlir::{attention_forward,grouped_matmul_tma_kernel}`：本文用它回答 attention 与 grouped-GEMM-shaped loop 是否能走完整 AutomaticWS transformation。
- `test/TritonGPU/partition-scheduling.mlir::{matmul_nested_persistent_ws_kernel,attention_persistent_inner_loop_kernel,scaled_mma_with_loads,scaled_mma_descriptor_scales}`：本文用它回答 persistent、nested attention 与 scaled MMA 的 partition outputs。
- `test/NVWS/aref-tmem-insertion.mlir::{nested_loop_yes_double_buffer,nested_loop_no_double_buffer,nested_loop_yes_double_buffer_scaled,nested_loop_no_double_buffer_scaled}`：本文用它回答 nested loop 是否满足 accumulator double-buffer 的证明条件。
- `test/Conversion/tritongpu_to_llvm_blackwell.mlir::{tc_gen5_mma_block_scale,tc_gen5_mma_block_scale_fp4_a,tc_gen5_mma_scaled_fp4_padded_tmem_lhs}`：本文用它回答 scaled case 最终需要哪些 TCGen05 instruction descriptors 与 scale operands。

外部来源：

- [triton-lang/triton#6239](https://github.com/triton-lang/triton/pull/6239)：本文用它回答 persistent matmul 为什么需要跨外层 tile loop 保持 scheduler 与 accumulator 状态。
- [triton-lang/triton#8236](https://github.com/triton-lang/triton/pull/8236)：本文用它回答 scale operand 无法与 MMA 一起安全 pipeline 时为什么需要保守同步。
- [triton-lang/triton#8687](https://github.com/triton-lang/triton/pull/8687)：本文用它回答 nested-loop partition propagation 以及基于执行证明的 TMEM hoist 条件。
- [triton-lang/triton#8883](https://github.com/triton-lang/triton/pull/8883)：本文用它回答 attention row-max 等 block-argument producer 的 stage/cluster provenance。
- [triton-lang/triton#10191](https://github.com/triton-lang/triton/pull/10191)：本文用它回答 scaled-MMA token 为什么必须作为 partition dataflow seed 传播到 TMEM consumer。

## Verifier、sanitizer 与成熟度边界

冻结源码与测试：

- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp::{VerifyWarpSpecializationPartitions,clearInternalWarpSpecializationAttrs,AutomaticWarpSpecialization::runOnOperation}`：本文用它回答为什么从 PartitionScheduling 到 LowerAref 的选定变换后验证 partition state，并在后续物化结束时清除分析属性。
- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/Partition.cpp::{verifyPartitionAttrs,verifyPartitionedLoop}`：本文用它回答 partition IDs、outputs、stages 与 warp-specialize tag 的局部一致性。
- `lib/Dialect/TritonInstrument/Transforms/ConcurrencySanitizer.cpp::ConcurrencySanitizerImpl`、`lib/Dialect/TritonNvidiaGPU/Transforms/ConSanNVIDIA.cpp::NVIDIAConSanHooks`：本文用它回答 ConSan 能建模哪些 WS/TMA/TCGen05/shared/TMEM memory effects 与 barrier state。
- `test/TritonGPU/partition-verifier-locality.mlir`：本文用它回答 internal attributes 只在消费它们的 pass boundary 才触发专用 verifier。
- `test/TritonGPU/automatic-warp-specialization.mlir::CLEAN` checks：本文用它回答 AutomaticWS 返回后是否仍泄漏 `ttg.partition`、partition outputs/stages 或 internal tag。
- `test/TritonGPU/consan.mlir`、`test/TritonGPU/consan-capture-reservation.mlir`：本文用它回答 sanitizer 对 captures、barrier lifecycle、proxy state、TMA completion 与 TCGen05 access 的 compile-time coverage。

外部来源：

- [PyTorch：Warp Specialization in Triton — Design and Roadmap](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/)：本文用它回答 AutomaticWS 的官方阶段划分、heuristic scheduler 定位，以及哪些能力仍被明确视为 generality、stability 与 tooling 工作。
- [triton-lang/triton#8189](https://github.com/triton-lang/triton/pull/8189)：本文用它回答 ConSan 为什么必须理解 partition scope、captures 与 asynchronous completion，而不能把 WS 当作普通 control flow。
- [triton-lang/triton#9212](https://github.com/triton-lang/triton/pull/9212)：本文用它回答 unsupported invariant 为什么应在 mutation 前导致安全 fallback，而不是在 TMEM ARef 后续阶段 assert。
- [triton-lang/triton#10058](https://github.com/triton-lang/triton/pull/10058)：本文用它回答 partition attrs 为什么必须是 pass-local scratch state，并在清理前持续验证。
- [triton-lang/triton#11067](https://github.com/triton-lang/triton/pull/11067)：本文用它回答截至冻结提交仍需 hardening 的 rewrite 稳健性类型，以及为什么“已有 canonical case”不能写成“任意 loop 已成熟”。

## 证据使用边界

- 生成 SM103 TTGIR、LLVM IR 或 PTX，只能证明目标结构与指令选择被构造；本文不据此声称运行正确、不会死锁或具有任何执行收益。
- GitHub PR/issue 的讨论用于回答设计理由、bug 触发条件和维护者判断；若其描述与冻结树不同，以冻结源码和测试为准。
- 官方 ISA 文档用于解释 lowering 必须满足的硬件契约；它不证明某个 Triton heuristic 对任意输入都能找到合法或理想分区。
- Tawa 用于解释 ARef 与 task-aware partitioning 的概念来源；Triton 当前实现的精确 pass order、operation set 和支持范围仍由冻结树决定。
- public API 文档、内部 lit coverage 与 roadmap 是三种不同证据：分别代表公开承诺、已防回归的实现范围和未完成方向，本文不将三者混写。
