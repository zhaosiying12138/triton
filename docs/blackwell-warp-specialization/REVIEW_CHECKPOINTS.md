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

正文机制章节按问题组织，不插 PR 编号或版本史；外部来源编号集中在本审查表与正文最后
的参考资料章。`research/github-discussion-audit.md` 和机器 ledger 作为语料审计工件，是这条
行文规则的有意例外。

| # | 正文知识点 | 主要冻结源码 | 正向/负向测试与学习 case | 外部材料组 | 状态 |
|---:|---|---|---|---|---|
| 1 | SM103/SM120 目标取舍；终局拓扑；TCGen05 alloc/MMA/commit/fence/ld-st-wait 生命周期与 `.mma.ws` 消歧；TMA、TMEM 的 issue/execute/storage 分工 | `AccelerateMatmul.cpp::getMMAVersionSafe`；`TargetFeatures.h`；`compiler.py::make_ttgir/make_llir`；`MMAv5.cpp`；`TensorMemoryToLLVM.cpp` | `tritongpu_to_llvm_sm120.mlir`；`tritongpu_to_llvm_blackwell.mlir`；Case 1；exact-SM103 PTX/SASS | NVIDIA compute capabilities/TMA/TMEM/TCGen05；GH-ISSUE-4308 | PASS A–G |
| 2 | NVWS dialect 的定位、2 个类型、属性/接口、16 个 op、legacy/current 两套 channel、四类 token/状态、6 个 pass、两个 transient epoch、完整 erase/TTGIR/LLVM 出口与 verifier/fail-safe 边界 | `NVWS/{IR,Transforms}` 全目录；`AutomaticWarpSpecialization.cpp`；旧 `WSCodePartition.cpp/WSLowerToken.cpp`；正式 TTG/LLVM converters | `test/NVWS/{ops,invalid,assign_stage_phase,lower_aref,lower_warp_group}.mlir`；`partition-loops.mlir`；Case 1B | GH-PR-6316/6359/6410/6728/7561/7611/7645/7581/7826/8009；96-URL cutoff audit | PASS A–G |
| 3 | 七类同步；所有相关 proxy/tensormap/mbarrier-init/TCGen05 fences；completion、ordering、visibility、ownership、rendezvous 的边界；ConSan 的动态 live roles、proxy frontier 与 partition-scoped cluster visibility | `LowerAref.cpp`；`BarrierOpToLLVM.cpp`；`TMAToLLVM.cpp`；`ProxyFenceInsertion.cpp`；`TMemBarrierInsertion.cpp`；`ClusterBarrierInsertion.cpp`；ConSan hooks | `consan.mlir`；`consan-capture-reservation.mlir`；`fence-inserstion.mlir`；`membar-cluster.mlir`；exact PTX/SASS 中 `after_thread_sync` 负向审计 | NVIDIA PTX barriers/fences；GH-PR-7278/8189/9456/9591/10192/10668/10864 | PASS A–G |
| 4 | 显式 `ttg.warp_specialize` 的 concurrent-region ABI、worker 无 SSA return、nested-WS 禁止与 verifier | `TritonGPUOps.td`；`Ops.cpp`；`LowerWarpGroup.cpp` | `ops.mlir`；`invalid.mlir`；`warp_specialize_to_llvm.mlir`；Case 2 | GH-PR-5917/5968 | PASS A–G |
| 5 | 普通 SWP、stage/cluster、分区后 single-stage normalization、nested marker、MMA RMW 与 self-latency、mixed TMA/non-TMA wait、descriptor `numStages+1` | `AssignLatencies.cpp`；`MMAv5PipelineUtility.cpp`；`ScheduleLoops.cpp`；`LowerLoops.cpp`；`multiBufferTMADescriptors` | `loop-pipeline-blackwell.mlir`；`pipeline-assign-latencies.mlir`；`pipeline-schedule-loop.mlir`；`pipeline-lower-loop.mlir`；mixed AutomaticWS；Case 3/4 | GH-PR-6887/6969/6984/8451/8883/9111 | PASS A–G |
| 6 | 前端请求、eligibility gate 与 pointer-only no-op contract | `core.py::range`；`code_generator.py::visit_For`；`hasEligibleMemoryOps` | `no_eligible_memory_ops`；`mma_no_memory_ops`；Case 4 | Triton `tl.range`；GH-PR-6217/9716 | PASS A–G |
| 7 | canonical TMA→TCGen05 自动 WS 的全链路物化 | `AutomaticWarpSpecialization.cpp` 及全部 NVWS subpasses | `matmul_tma_ws_kernel`；`warp_specialize_tma_matmul`；Case 5；`lab/tma_matmul` | GH-PR-6217/8262 | PASS A–G |
| 8 | PartitionScheduling graph、data propagation、merge/cost、cheap/multi-use cloning；for/if/单结果 reduce 的 generic-CF 实际边界 | `PartitionScheduling.cpp`；`PartitionSchedulingUtility.cpp`；`Partition.cpp`；`PartitionLoops.cpp` | `partition-scheduling.mlir`；`partition-loops.mlir`；`partition-verifier-locality.mlir`；Case 6 | GH-PR-6175/7312/7415/11067 | PASS A–G |
| 9 | Shared ARef 的 channel、last consumer、stage/phase 与 LowerAref | `InsertAref.cpp`；`LowerAref.cpp`；`AssignStagePhase.cpp` | `insert_aref.mlir`；`lower_aref.mlir`；Case 7 | Tawa；GH-PR-6288/7479/7645/9114 | PASS A–G |
| 10 | TMEM ARef 的 token DAG、partition-keyed async kind、no-marker 早退、two-owner lowering 中心与 accumulator double buffer；显式 IDs 超过两个直接 assert，root+两个显式 owners 只越过该断言、并未证明三-owner 支持 | `InsertTmemAref.cpp`；`HoistTmemStore.cpp` | `aref-tmem-insertion.mlir`；`tmem_barrier_insertion.mlir`；Case 8 | GH-PR-9007/9212 | PASS A–G |
| 11 | TTIR→TTGIR→LLVM 的精确 pass 顺序、逐步 verifier/cleanup，以及明确关闭的 integer-range optimization | `compiler.py::make_ttgir/make_llir`；`AutomaticWarpSpecialization.cpp` | `automatic-warp-specialization.mlir` 三组 RUN/CLEAN | PyTorch WS roadmap；GH-PR-6335/6378/10058 | PASS A–G |
| 12 | logical partitions 到 physical warpgroups；寄存器 pool、`setmaxnreg` 与 TMEM start-ID 对齐 FIXME | `OptimizePartitionWarps.cpp`；`AllocateWarpGroups.cpp`；`createRegRealloc` | `optimize-partition-warps.mlir`；`allocate_warp_groups.mlir`；Case 9 | NVIDIA `setmaxnreg`；GH-PR-6323/6407/6694/8005 | PASS A–G |
| 13 | persistent worker switch、capture ABI/remat、relative warp id 与 LICM | `WarpSpecializeUtility.cpp`；`ConvertWarpSpecializeToLLVM.cpp` | `warp_specialize_to_llvm.mlir`；Case 10；`lab/ws_skeleton` | GH-PR-5968/6694 | PASS A–G |
| 14 | scaled/blockscale 的 scale channel、dependency/mod-ref token、独立 completion protocol 与目标指令组合 | `PartitionScheduling::initialDataValues`；`InsertTmemAref.cpp`；`MMAv5.cpp` | scaled partition/ARef/PTX fixtures；Case 11 | GH-PR-8236/10191 | PASS A–G |
| 15 | 2CTA TCGen05、cross-CTA hazard、双 mbarrier slot 与 all-warps init | `ClusterBarrierInsertion.cpp`；`ClusterBarrierMbarAllocator.cpp`；`ClusterOpsToLLVM.cpp` | `tc_gen5_mma_2ctas`；cluster allocator/WS tests；Case 12 | NVIDIA PTX cluster semantics；GH-PR-9456/10914 | PASS A–G |
| 16 | persistent GEMM/attention 与 grouped GEMM 的 nested state、TMEM 生命周期、动态 descriptor | `HoistTmemStore.cpp`；`PartitionLoops.cpp`；`multiBufferTMADescriptors` | persistent/grouped Python 与 MLIR fixtures；Cases 13–14 | GH-PR-6239/8687/8883 | PASS A–G |
| 17 | case 支持矩阵、明确的整体成熟度判断，以及 no-op/diagnostic/assert/FIXME 四层 fail-safe 矩阵 | 上述完整链；FIXME/NYI/assert/fallback sites | 14-case matrix；focused lit 目标集；本地精选证据 | PyTorch roadmap；GH-PR-8189/9212/10058/10192/10668/10864/11067 | PASS A–G |

## 2026-07-30 缺口回补复核

| 审计缺口 | 正文落点 | 复核结果 |
|---|---|---|
| NVWS dialect 的完整语言边界与两个 erase epoch | 3.1–3.9、Case 1B | PASS：类型/属性/接口/16 ops/6 passes/lowering/fail-safe 已写；补齐 legacy 五-op 状态流、ArefBuffer/warp-group verifier holes、CLEAN 证据边界与 reverted→re-landed provenance；focused runner 31/31 files、43 RUN |
| same-stage normalization、nested marker、RMW、fully-WS/mixed self-latency | 6.2–6.5、Case 3、Case 4 | PASS：源码调用关系、四组 focused fixtures 与设计动机已串成同一因果链 |
| explicit nested WS、worker return、generic CF、integer-range、TMEM alignment | 5、9.1、12.2、13.2 | PASS：区分 verifier 边界、assert/fatal 边界、主动关闭优化与未证明 FIXME |
| #4308/#9007/#9111/#9212/#11067 的机制动机 | 6、11.2/11.3、9.6、19.5 | PASS：动机放回相应机制，不改写成 PR 时间线 |
| correctness hardening #10192/#10668/#10864 | 4.9、19.4、参考索引 | PASS：作为 sanitizer/maturity 证据，不冒充硬件资格化 |
| 当前成熟度到底如何 | 19.4 | PASS：给出 scoped verdict，不用单一分数掩盖证据轴差异 |
| fail-safe 保护层 | 19.5 | PASS：安全 no-op、明确 diagnostic、assert/unreachable、无负例源码边界逐项分开 |

## 2026-07-30 最终交付门禁

| 门禁 | 结果 |
|---|---|
| 正文 `path::symbol` 与冻结源码/测试逐项抽查 | PASS；独立 symbol audit 无 P0/P1，组合 selector 与描述性 selector 已明确区分 |
| 原计划 17 个知识 checkpoint | PASS A–G；两轮独立正文审计最终均为无 P0、无 P1 |
| 14 个 machine case 与正文教学切片映射 | PASS；§19.2 每个稳定 ID 恰好出现一次，Case 6 明确为共用机制切片 |
| 外部链接与 GitHub 语料覆盖 | PASS；正文外链只在第 21 章，137 个正文 URL 全被 ledger 覆盖；广域 audit 与 index 均为 96 个 GitHub URL |
| JSON、Markdown 本地链接、HTML anchors/标题 | PASS；5 个核心 JSON 可解析，本地链接无缺失，HTML 为 21 个二级章节且无坏 fragment；`build_book.py --check` 通过 |
| focused compiler tests | PASS；`-j24` 执行 31/31 files，覆盖 43 条 `RUN:`，包含 `test/NVWS` 全部 8 个文件 |
| exact-SM103 compile-only evidence | PASS；新 manifest hash 下 4/4 records 通过，published evidence 同 hash；未加载或启动 cubin |
| 运行时与性能措辞 | PASS；没有把 compile structure、仓库 GPU test source、lit wall time 或 SASS 当成硅上结果；缺失的 SM103 数值、deadlock、occupancy、profiling 与 speedup 项保留在 deferred test plan |
