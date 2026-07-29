# Warp Specialization 写作审查清单

这张表约束正文的完成标准。一个 checkpoint 只有同时通过以下七项才可标记为
`PASS`：

- **A 源码**：在冻结提交中重读实现与相邻 pass，不只依赖符号搜索；
- **B 仓库测试**：找到正例、边界或负例，并确认 FileCheck 实际检查什么；
- **C 学习 case**：把知识点落到正文 case 或 `lab/` 中的可复现输入；
- **D 现象**：说明变换前后应观察的 IR/PTX/SASS 结构及其因果含义；
- **E 外部材料**：吸收 `source-ledger.json` 中对应 issue、PR、ISA 或论文的设计动机；
- **F 边界**：明确 heuristic、未实现形状及 compile-only 不能证明的结论；
- **G 行文**：人工通读，删除流水账、空泛结论、硬译术语和重复铺垫。

外部来源的编号只在本审查表和正文最后的参考资料章出现。正文知识点本身按问题组织，
不写版本史。

| # | 正文知识点 | 主要冻结源码 | 正向/负向测试与学习 case | 外部材料组 | 状态 |
|---:|---|---|---|---|---|
| 1 | SM103/SM120 目标取舍；终局拓扑；TCGen05 alloc/MMA/commit/fence/ld-st-wait 生命周期与 `.mma.ws` 消歧；TMA、TMEM 的 issue/execute/storage 分工 | `AccelerateMatmul.cpp::getMMAVersionSafe`；`TargetFeatures.h`；`compiler.py::make_ttgir/make_llir`；`MMAv5.cpp`；`TensorMemoryToLLVM.cpp` | `tritongpu_to_llvm_sm120.mlir`；`tritongpu_to_llvm_blackwell.mlir`；Case 1；exact-SM103 PTX/SASS | NVIDIA compute capabilities/TMA/TMEM/TCGen05；GH-ISSUE-4308 | PASS A–G |
| 2 | 七类同步；所有相关 proxy/tensormap/mbarrier-init/TCGen05 fences；completion、ordering、visibility、ownership、rendezvous 的边界；ConSan 可见范围 | `LowerAref.cpp`；`BarrierOpToLLVM.cpp`；`TMAToLLVM.cpp`；`ProxyFenceInsertion.cpp`；`TMemBarrierInsertion.cpp`；`ClusterBarrierInsertion.cpp`；ConSan hooks | `consan.mlir`；`fence-inserstion.mlir`；`membar-cluster.mlir`；exact PTX/SASS 中 `after_thread_sync` 负向审计 | NVIDIA PTX barriers/fences；GH-PR-7278/8189/9456/9591 | PASS A–G |
| 3 | 显式 `ttg.warp_specialize` 的 concurrent-region ABI 与 verifier | `TritonGPUOps.td`；`Ops.cpp`；`LowerWarpGroup.cpp` | `ops.mlir`；`invalid.mlir`；`warp_specialize_to_llvm.mlir`；Case 2 | GH-PR-5917/5968 | PASS A–G |
| 4 | 普通 software pipeline、stage/cluster 与 descriptor `numStages+1` | `AssignLatencies.cpp`；`ScheduleLoops.cpp`；`LowerLoops.cpp`；`multiBufferTMADescriptors` | `loop-pipeline-blackwell.mlir`；`pipeline-assign-latencies.mlir`；Case 3 | GH-PR-6887/8883/9111 | PASS A–G |
| 5 | 前端请求、eligibility gate 与 pointer-only no-op contract | `core.py::range`；`code_generator.py::visit_For`；`hasEligibleMemoryOps` | `no_eligible_memory_ops`；`mma_no_memory_ops`；Case 4 | Triton `tl.range`；GH-PR-6217/9716 | PASS A–G |
| 6 | canonical TMA→TCGen05 自动 WS 的全链路物化 | `AutomaticWarpSpecialization.cpp` 及全部 NVWS subpasses | `matmul_tma_ws_kernel`；`warp_specialize_tma_matmul`；Case 5；`lab/tma_matmul` | GH-PR-6217/8262 | PASS A–G |
| 7 | PartitionScheduling graph、data propagation、merge/cost 与 cheap cloning | `PartitionScheduling.cpp`；`PartitionSchedulingUtility.cpp`；`Partition.cpp` | `partition-scheduling.mlir`；`partition-verifier-locality.mlir`；Case 6 | GH-PR-6175/7312/7415/11067 | PASS A–G |
| 8 | Shared ARef 的 channel、last consumer、stage/phase 与 LowerAref | `InsertAref.cpp`；`LowerAref.cpp`；`AssignStagePhase.cpp` | `insert_aref.mlir`；`lower_aref.mlir`；Case 7 | Tawa；GH-PR-6288/7479/7645/9114 | PASS A–G |
| 9 | TMEM ARef 的 token DAG、至多两个显式 partition IDs（root owner 另计）与 accumulator double buffer | `InsertTmemAref.cpp`；`HoistTmemStore.cpp` | `aref-tmem-insertion.mlir`；`tmem_barrier_insertion.mlir`；Case 8 | GH-PR-9007/9212 | PASS A–G |
| 10 | TTIR→TTGIR→LLVM 的精确 pass 顺序、逐步 verifier 与 cleanup | `compiler.py::make_ttgir/make_llir`；`AutomaticWarpSpecialization.cpp` | `automatic-warp-specialization.mlir` 三组 RUN/CLEAN | PyTorch WS roadmap；GH-PR-10058 | PASS A–G |
| 11 | logical partitions 到 physical warpgroups；寄存器 pool 与 `setmaxnreg` | `OptimizePartitionWarps.cpp`；`AllocateWarpGroups.cpp`；`createRegRealloc` | `optimize-partition-warps.mlir`；`allocate_warp_groups.mlir`；Case 9 | NVIDIA `setmaxnreg`；GH-PR-6323/6407/6694/8005 | PASS A–G |
| 12 | persistent worker switch、capture ABI/remat、relative warp id 与 LICM | `WarpSpecializeUtility.cpp`；`ConvertWarpSpecializeToLLVM.cpp` | `warp_specialize_to_llvm.mlir`；Case 10；`lab/ws_skeleton` | GH-PR-5968/6694 | PASS A–G |
| 13 | scaled/blockscale 的 scale channel、dependency/mod-ref token、独立 completion protocol 与目标指令组合 | `PartitionScheduling::initialDataValues`；`InsertTmemAref.cpp`；`MMAv5.cpp` | scaled partition/ARef/PTX fixtures；Case 11 | GH-PR-8236/10191 | PASS A–G |
| 14 | 2CTA TCGen05、cross-CTA hazard、双 mbarrier slot 与 all-warps init | `ClusterBarrierInsertion.cpp`；`ClusterBarrierMbarAllocator.cpp`；`ClusterOpsToLLVM.cpp` | `tc_gen5_mma_2ctas`；cluster allocator/WS tests；Case 12 | NVIDIA PTX cluster semantics；GH-PR-9456/10914 | PASS A–G |
| 15 | persistent GEMM/attention 与 grouped GEMM 的 nested state、TMEM 生命周期、动态 descriptor | `HoistTmemStore.cpp`；`PartitionLoops.cpp`；`multiBufferTMADescriptors` | persistent/grouped Python 与 MLIR fixtures；Cases 13–14 | GH-PR-6239/8687/8883 | PASS A–G |
| 16 | case 支持矩阵、机制成熟度与可证/不可证边界 | 上述完整链；FIXME/NYI/assert/fallback sites | 13-case matrix；目标 lit 集；本地精选证据 | PyTorch roadmap；GH-PR-8189/9212/10058/11067 | PASS A–G |

最终审查还必须确认：正文中的每个 `path::symbol` 在冻结树中存在；所有外部 URL 只出现在
最后参考资料章；Markdown 与 HTML 标题顺序一致；实验记录没有 kernel launch；全文没有把
编译结构、仓库中的硬件测试源码或 SASS 反汇编误写成运行结果。
