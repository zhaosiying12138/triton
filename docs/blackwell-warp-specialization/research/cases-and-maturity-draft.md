# Case 与成熟度研究稿（已收口）

这份路径保留为研究过程入口；原先从 persistent dispatch 开始的局部章节稿已经逐段吸收到 `../article.md` 第 14–19 章。为避免旧章节号、旧 owner 上限和“只给向量、不作判断”的阶段性措辞继续误导，下面只保存最终路由与结论。

## 1. 十四个可机读 case

权威定义在 `case-matrix.json`，正文按依赖关系由浅入深展开：

1. `explicit_ws_micro_ir`：显式 TTGIR default/worker ABI 与 persistent switch loop；
2. `explicit_tmem_micro_ir`：显式 WS 下的 TMEM allocation/capture/lowering；
3. `nvws_dialect_lifecycle`：ARef epoch 与 warp-group epoch 的创建、擦除和正式 TTGIR 出口；
4. `ordinary_async_no_auto_ws`：有 marker、无 eligible descriptor memory 时安全 no-op；
5. `mixed_tma_async_pipeline`：TMA 与 ordinary async-copy 两条不对称 operand chains；
6. `tma_auto_ws_gemm`：canonical TMA→TCGen05→TMEM AutomaticWS；
7. `persistent_tcgen05_gemm`：nested persistent loop 与 TMEM initialization hoist；
8. `shared_aref`：Shared/register payload 的 put/get channel；
9. `tmem_aref`：TMEM RAW/WAR/lifetime ownership hand-off；
10. `register_allocation`：logical partitions、physical warpgroups 与 `setmaxnreg`；
11. `scaled_blockscale`：scale provenance、TMEM scale channel 与同步/异步选择；
12. `two_cta_cluster`：2CTA 的组件级证据束与 slot/parity 共同快照；
13. `persistent_attention`：nested state、两条 accumulator owner chains 与输出分类；
14. `grouped_gemm`：动态 descriptor lifetime 与 `numStages + 1` buffering。

## 2. 证据轴不能相互代偿

每个 case 分别记录 frontend/explicit input、automatic discovery、focused transform、composed pipeline、target lowering 和 runtime qualification。Python 源码存在不证明同名 TTGIR 的生成 provenance；focused FileCheck 不证明同一输入贯穿全部 passes；PTX/cubin/SASS 结构不证明 launch、数值、deadlock-free、occupancy 或 speedup。

本项目唯一硬件结论是选择 exact SM103 compile-only。SM120 会走 MMA v2 且缺少本文所需的同一 cluster/TCGen05 路径，不能拿 SM120 runtime 代替 SM103 资格化。设备实验项保存在 `../lab/TEST_PLAN.md`，等待有权限且有合适 SM103 硬件后补测。

## 3. 最终成熟度判断

冻结树中的显式 WS IR、persistent lowering、warp/register allocation 集成程度高；canonical simple TMA GEMM 是 compile-time 证据链最完整的中心 case。Automatic partition policy 仍是 loop-local heuristic，不是全局任务调度器；scaled、2CTA、attention、persistent 与 grouped GEMM 有分层支持，但端到端 provenance 深度不同。ConSan、async proxy、cluster visibility 和 unusual control-flow/tile rewrites 仍持续加固，而本项目没有 SM103 launch、数值、deadlock、occupancy 或性能记录。

因此最终定性是：**核心编译机制已经集成，canonical TMA GEMM 的 compile-time 路径较成熟；通用 workload、自动策略的全局性与 production runtime qualification 仍偏实验性。**

## 4. Fail-safe 不能压成一个标签

正文第 19.5 节与 `source-map.json::kp_fail_safe_boundaries` 分开记录四层行为：

- 安全 no-op：无 eligible memory、序列化后至多一个 partition、函数没有 WS marker；
- 明确 diagnostic/pass failure：显式 WS verifier、TMEM subview、未消除 direct SSA 等；
- assert/unreachable/fatal 或潜在 deadlock：过多显式 TMEM owners、跨 partition cp.async、未支持 async kind，以及手写 NVWS 的空 backing/空 completion/token pairing 边界；
- 源码 FIXME 或尚无负例：TMEM warpgroup start-ID alignment、root 加两个 explicit TMEM owners、generic region control flow 等。

NVWS parser/verifier 的可接受域还大于已证明 lowering 域：zero-region/no-result `nvws.warp_group` 能 round-trip，但 lowering 会索引第一个 group；ARef verifier 也不证明 non-empty、unique ownership 或完整 transaction pairing。这些属于成熟度证据，不能改写成常见 canonical case 已失败。
