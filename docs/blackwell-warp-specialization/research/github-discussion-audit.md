# Blackwell Warp Specialization GitHub 讨论审计

> 冻结源码：`bf64a5db1bc8aab0fd4f0076e60f6c367852e47d`
>
> 审计截止：2026-07-30
>
> 目的：说明本文实际检索了什么、哪些材料进入了哪个知识点、哪些只用于成熟度判断，以及“完整”能够诚实地指到哪里。

## 1. 可主张的覆盖范围

本文做的是以下四路交叉审计：

1. GitHub issue/PR 搜索，范围包含 title、body 与可索引 comments；
2. 可访问的 issue comments、PR review comments 与 HTML discussion；
3. 本地完整 git history 中的 subject/body/PR 编号；
4. 对冻结版本 WS/NVWS 核心实现做 blame，反查当前仍存活代码行的来源。

它支持这样的表述：

> 对冻结源码当前实现行、GitHub 可检索的 issue/PR/comments/reviews，以及关联 commits 做了系统性审计，并把直接改变机制解释或成熟度判断的材料逐项归档。

它不支持这样的表述：

> 数学意义上枚举了 GitHub 内外所有曾经出现过、后来被编辑或删除的 WS 讨论。

GitHub 搜索存在索引延迟、排序与最多 1000 条结果的限制；被删除/编辑的 review、force-push 后不可达的旧 diff、branch-only 讨论、Slack/内部 Meta/NVIDIA 讨论无法恢复。关键词也必然有边界：部分提交只写 `partition`、`ARef`、`mbarrier` 或具体 bug，并不写 “warp specialization”。

## 2. 搜索面与规模

| Query | 命中数 | 主要用途 |
|---|---:|---|
| `"warp specialization" in:title,body,comments` | 185 | 主语料池 |
| `warp_specialize` | 46 | IR/API 拼写 |
| `NVWS` | 47 | dialect、ARef、legacy token |
| `PartitionScheduling` | 10 | 自动分区策略 |
| `PartitionLoops` | 5 | materialization/control flow |
| `InsertAref` / `LowerAref` | 8 / 9 | Shared ARef 生命周期 |
| `InsertTmemAref` | 5 | TMEM ownership |
| `LowerWarpGroup` | 2 | NVWS→正式 TTGIR |
| `"ttg.partition"` | 12 | 临时分析 ABI |
| `setmaxnreg` / `"register reallocation"` | 2 / 3 | 动态寄存器协议 |
| `"warp specialized"` | 83 | 未使用名词形式的讨论 |
| `AutomaticWarpSpecialization` | 3 | composite pass |
| `ConSan warp` | 29 | sanitizer/tooling |

`source-ledger.json` 在本轮前有 52 个逐项记录的 URL，其中 42 个是 Triton GitHub issue/PR。宽搜索与本地 history 的简单差集会产生 122 个额外编号；其中大量属于 Hopper code partition、AMD、Gluon、release backport、通用 sanitizer 或仅相邻重构，不能把 122 误写成“122 个都属于冻结 Blackwell AutomaticWS”。对冻结实现行做 blame 后，约 35 组 ledger 外来源对当前机制具有直接 provenance；其余按“历史路线”“成熟度问题”“邻接/排除”分类。收口后的 ledger 仍保留原有逐项炼化 records，同时在 `github_discussion_audit.indexed_github_urls` 中登记 96 个广域 GitHub URL；后者是发现索引，不冒充 96 份同等深度的 source record。

## 3. 进入正文机制解释的新增材料

下面不是 PR 时间线，而是“当前代码为什么这样写”的来源映射。正文用问题组织；链接只在文章最终参考章出现。

### 3.1 NVWS、ARef 与 ownership

| 材料 | 从 discussion/review 提炼出的机制 | 正文落点 |
|---|---|---|
| [#6316](https://github.com/triton-lang/triton/pull/6316)、[#6359](https://github.com/triton-lang/triton/pull/6359) | `nvws.warp_group` 是 progressive-lowering container；随后归一为正式 `ttg.warp_specialize`，不是第三套最终 ABI | §3.3、§3.7 |
| [#6410](https://github.com/triton-lang/triton/pull/6410) | 初代 ARef 将抽象生命周期降成 SMEM + barrier protocol | §3、§10 |
| [#6520](https://github.com/triton-lang/triton/pull/6520) | TMEM token 类似 MemorySSA ownership edge；不仅是 RAW，也承载 WAR/lifetime | §3.4、§11 |
| [#6728](https://github.com/triton-lang/triton/pull/6728) | 从 region/rewrite-multiplicity ARef 迁移到 put/get enter/exit，使 lifetime 可配对、可渐进 lowering | §3.1、§3.3 |
| [#7561](https://github.com/triton-lang/triton/pull/7561)、[#7611](https://github.com/triton-lang/triton/pull/7611)、[#7645](https://github.com/triton-lang/triton/pull/7645) | enter/exit 生命周期先由 #7561 落地、随即由 #7611 回退，再由 #7645 的 take-two 重新落地；当前冻结 InsertAref 还叠加后续重构，因此动机可继承，具体实现不能归因给被回退版本 | §3、§10 |
| [#7581](https://github.com/triton-lang/triton/pull/7581) | 当前 InsertAref 核心；first/last user 应按 stage/cluster pipeline time，而非文本顺序 | §3.5、§10 |
| [#7590](https://github.com/triton-lang/triton/pull/7590) | `is_async` 必须是显式语义，不能从 barrier operands 的形状反推 | §3.6、§6、§10 |
| [#7648](https://github.com/triton-lang/triton/pull/7648)、[#7649](https://github.com/triton-lang/triton/pull/7649) | 多 consumer；enter/exit 自身需要 stage/cluster metadata | §3、§10 |
| [#7686](https://github.com/triton-lang/triton/pull/7686) | generic cross-partition SSA ARef 目前固定 depth=1，是实现边界，不是通用多缓冲承诺 | §10、§19 |
| [#7757](https://github.com/triton-lang/triton/pull/7757) | lifetime tag/final wait 必须归属真正完成生命周期的 partition | §10、§12 |
| [#7826](https://github.com/triton-lang/triton/pull/7826) | 当前 active TMA ARef；channel combine/multibuffer 只对 TMA producer 启用 | §3.6、§10 |
| [#7927](https://github.com/triton-lang/triton/pull/7927) | AssignStagePhase 与 LowerAref 分层，明确 schedule decision 先于 concrete barrier rewrite | §3.5、§12 |
| [#8009](https://github.com/triton-lang/triton/pull/8009) | TMEM access DAG、loop entry/exit owner transition、两 owner ping-pong | §3、§11 |
| [#8101](https://github.com/triton-lang/triton/pull/8101)、[#8188](https://github.com/triton-lang/triton/pull/8188)、[#8329](https://github.com/triton-lang/triton/pull/8329) | stage/phase 先成为 optional，再只传播给真正需要的 partition；外部 scalar dependency 归入 default | §3.2、§3.5、§10 |
| [#8197](https://github.com/triton-lang/triton/pull/8197) | descriptor load 与 local alloc 在不同 partition 时可能需要两段 ownership hand-off | §10 |
| [#8619](https://github.com/triton-lang/triton/pull/8619) | dependency rewrite 最终折回 InsertAref，Shared payload 与 TMEM ownership 保持不同模型 | §3、§10、§11 |

### 3.2 partition graph、control flow 与 MLIR analysis ABI

| 材料 | 提炼出的机制 | 正文落点 |
|---|---|---|
| [#6597](https://github.com/triton-lang/triton/pull/6597) | 从固定 producer/consumer 扩展到多个 load groups、多个 MMA；assignment 与 communication lowering 分离 | §7–§9 |
| [#6660](https://github.com/triton-lang/triton/pull/6660) | attention 把 vector/SFU work 独立成 partition；PTXAS 不会替前端跨 barrier 搬代码 | §9、§17 |
| [#6876](https://github.com/triton-lang/triton/pull/6876) | PartitionScheduling 独立成 pass，便于把 policy 与 materialization 分开测试 | §9、§12 |
| [#8123](https://github.com/triton-lang/triton/pull/8123) | 从隐式 root 转向 partition set；穷举 annotation 同时服务 lowering、验证和调试 | §9 |
| [#8215](https://github.com/triton-lang/triton/pull/8215)、[#8534](https://github.com/triton-lang/triton/pull/8534) | all-ops annotation 与 if/yield 处理，避免靠缺省 root 猜测 | §9、§19 |
| [#8634](https://github.com/triton-lang/triton/pull/8634)、[#8651](https://github.com/triton-lang/triton/pull/8651)、[#8656](https://github.com/triton-lang/triton/pull/8656) | loop control operands、WS loop 外部 op 与 then/else 对称 heuristic 的边界 | §9、§12 |
| [#8799](https://github.com/triton-lang/triton/pull/8799)、[#9023](https://github.com/triton-lang/triton/pull/9023)、[#9133](https://github.com/triton-lang/triton/pull/9133) | RegionBranchInterface、region arguments、capture holder 让 MLIR dataflow 能理解 WS regions | §5、§9 |
| [#10560](https://github.com/triton-lang/triton/pull/10560) | if split 后 metadata 必须从真实结构推导，不能硬编码 | §9、§19 |

### 3.3 software pipeline、latency 与 barrier placement

| 材料 | 提炼出的机制 | 正文落点 |
|---|---|---|
| [#6174](https://github.com/triton-lang/triton/pull/6174) | `ttng.arrive_barrier` 的引入；producer 需要显式 publish，而不是 consumer 只有 wait | §4 |
| [#6761](https://github.com/triton-lang/triton/pull/6761) | `tt.self_latency` 的原始动机：异步 MMA 对自身后继 iteration 的完成约束 | §6 |
| [#6917](https://github.com/triton-lang/triton/pull/6917) | critical path 固定在 default 侧；RMW accumulator 不应重复贡献 MMA latency | §6、§9 |
| [#7336](https://github.com/triton-lang/triton/pull/7336) | 单 warp participant 用 `bar.warp.sync`，不消耗同样的 CTA named-barrier 执行形态 | §4、§14 |
| [#8311](https://github.com/triton-lang/triton/issues/8311)、[#8423](https://github.com/triton-lang/triton/pull/8423) | instruction reorder 曾把 arrive 越过 local load，导致 silent wrong result；barrier placement 是 correctness | §4、§19 |
| [#8317](https://github.com/triton-lang/triton/pull/8317) | `local_store`→MMAv5 consumer 需要独立 proxy fence | §4、§10 |
| [#8415](https://github.com/triton-lang/triton/pull/8415) | 分区后第二次 ScheduleLoops 还防止 operation 被排到 operand 之前 | §6、§12 |
| [#8797](https://github.com/triton-lang/triton/pull/8797) | stage/cluster 尚不能原生按 partition 表达，冻结实现保留 workaround | §6、§19 |
| [#11014](https://github.com/triton-lang/triton/issues/11014) | 已 materialize 的 WS schedule 即使 `num_stages<=1` 也不能按普通 loop 规则丢弃 | §6 |

### 3.4 persistent worker、寄存器与 instrumentation

| 材料 | 提炼出的机制 | 正文落点 |
|---|---|---|
| [#5963](https://github.com/triton-lang/triton/pull/5963) | partition regions 在 runtime 并发存活，register interference 不能按文本顺序建模；同时引入 relative IDs/WS allocation analysis | §5、§13、§14 |
| [#6403](https://github.com/triton-lang/triton/pull/6403) | TMEM acquire 跨 epilogue 与 load-role warp 数会改变 register-pressure policy | §13 |
| [#6798](https://github.com/triton-lang/triton/pull/6798) | default/worker 间开始显式借用寄存器预算 | §13 |
| [#6870](https://github.com/triton-lang/triton/pull/6870) | switch worker loop 禁止 LLVM LICM，防止 case work 被外提到错误角色 | §14 |
| [#6877](https://github.com/triton-lang/triton/pull/6877) | worker 在 inactive 阶段立即归还寄存器，24-register floor 是协议的一部分 | §13、§14 |
| [#9786](https://github.com/triton-lang/triton/pull/9786) | sanitizer instrumentation 自身占寄存器，可能与动态 realloc 互锁；需要禁用通路 | §13、§19 |
| [#10417](https://github.com/triton-lang/triton/pull/10417) | assert/print/device runtime 需要额外最低寄存器预算 | §13 |

### 3.5 scaled、attention、persistent 与 multi-CTA

| 材料 | 提炼出的机制 | 正文落点 |
|---|---|---|
| [#6299](https://github.com/triton-lang/triton/pull/6299) | persistent WS 的 capture/remat/descriptor multibuffer；早期 waiter role 后来被移除 | §17、§18 |
| [#6514](https://github.com/triton-lang/triton/pull/6514) | 不再额外建立 waiter partition；MMAv5 completion 可以服务多个 consumer，FMHA 因而可表达 | §9、§10、§17 |
| [#6551](https://github.com/triton-lang/triton/pull/6551) | scale TMA optional 与 block-scaled WS 的早期组合边界 | §15 |
| [#7734](https://github.com/triton-lang/triton/pull/7734) | scale 从 TMEM→SMEM 可能被下一 iteration 覆盖；async load + sync MMA 是合法保守组合 | §11、§15 |
| [#8950](https://github.com/triton-lang/triton/pull/8950) | accumulator double buffering 是 producer-consumer topology 性质，不能数 owner changes 猜测 | §11、§15 |
| [#10014](https://github.com/triton-lang/triton/pull/10014) | descriptor interface 泛化到 gather/scatter；冻结树已有 `descriptor_gather`/async gather 编译证据 | §3、§7、§19 |
| [#10440](https://github.com/triton-lang/triton/pull/10440) | cluster barrier 进入 WS region 的直接来源 | §16 |
| [#10571](https://github.com/triton-lang/triton/pull/10571) | scaled case 用 LinearLayout 等价判断避免不必要的 SMEM hand-off | §15 |
| [#10656](https://github.com/triton-lang/triton/pull/10656)、[#10669](https://github.com/triton-lang/triton/pull/10669)、[#11010](https://github.com/triton-lang/triton/pull/11010) | multi-CTA exit、双 buffer 与 phase snapshot：slot 和 parity 必须共同快照 | §16 |

## 4. 只进入成熟度/限制的 open、draft 与问题报告

这些材料不能反向写成冻结提交“已支持”或“仍必现”；它们的价值是暴露适用域、历史 failure mode 与尚未闭合的设计工作。

| 状态 | 材料 | 对成熟度判断的含义 |
|---|---|---|
| open | [#7628](https://github.com/triton-lang/triton/issues/7628) | loop-carried ARef ownership 缺高效 phi 表达；现有机制可保守工作但通信/同步可能过多 |
| open | [#9039](https://github.com/triton-lang/triton/issues/9039) | per-partition `stage/cluster` 仍是架构债务 |
| open + fix draft | [#9853](https://github.com/triton-lang/triton/issues/9853)、[#10753](https://github.com/triton-lang/triton/pull/10753) | `if_op_result_token` heuristic 的 missing return 显示 policy 边缘仍在修补 |
| open / main 已 no-op | [#10284](https://github.com/triton-lang/triton/issues/10284) | sm120 没有投入正向 WS 支持；non-TMA pointer-load case 应正确 no-op，而不是作为性能 case |
| fixed chain | [#7354](https://github.com/triton-lang/triton/issues/7354) | `num_warps<4` 曾触发 allocator crash；当前 verifier/fix chain 说明资源边界不是天然安全 |
| fixed | [#8311](https://github.com/triton-lang/triton/issues/8311)、[#8423](https://github.com/triton-lang/triton/pull/8423) | barrier reorder 曾产生 silent wrong result，证明 correctness hardening 不能只看 compiler crash |
| fixed | [#8571](https://github.com/triton-lang/triton/issues/8571)、[#8534](https://github.com/triton-lang/triton/pull/8534) | yield/third-partition handoff 可表达，但 heuristic 与性能质量仍 shape-sensitive |
| fixed | [#8932](https://github.com/triton-lang/triton/issues/8932)、[#9007](https://github.com/triton-lang/triton/pull/9007) | 无 MMAv5 的 marked loop 曾继续进入 ARef；现在应在 mutation 前 no-op |
| fixed/环境不足 | [#8072](https://github.com/triton-lang/triton/issues/8072) | attention pass crash 有缩减修复，但当时缺 RTX 5090 端到端验证 |
| closed/nonrepro | [#10901](https://github.com/triton-lang/triton/issues/10901) | B200 partition attr 报告在新 main 不可复现；不能宣称冻结树仍复现 |
| closed/unmerged | [#9524](https://github.com/triton-lang/triton/pull/9524) | nested/small non-MMAv5 loop 的空 `partition.outputs` 是潜在路径，只能写未保护 accessor，不能写已复现 bug |
| closed/downstream | [#9597](https://github.com/triton-lang/triton/pull/9597) | metadata `num_warps` 与 total warps 的边界最终在下游修复，不计入冻结主线能力 |
| closed/unmerged | [#10195](https://github.com/triton-lang/triton/pull/10195) | epilogue peeling 未见收益，不能把未落地优化当成缺 correctness |
| draft redesign | [#10121](https://github.com/triton-lang/triton/pull/10121) | ARef→Semaphore 提案说明当前 enter/exit、stage/phase 与多 owner 模型仍在探索 |
| open | [#8969](https://github.com/triton-lang/triton/pull/8969) | per-warp arrival count 尚未进入冻结实现；当前 elected-thread arrival 是保守方案 |
| draft / frozen source 已对账 | [#8802](https://github.com/triton-lang/triton/pull/8802) | draft 记录该 flag 曾需单独补强；冻结 `InsertTmemAref::insertTmemAref` 已检查 `!getDisallowAccMultiBuffer(wsLoop)`，`nested_loop_no_double_buffer_scaled` 也检查 depth=1。它不再是 canonical direct-MMA path 的当前缺口；任意 nested/multi-MMA 组合仍不能由单一 fixture 外推 |
| draft | [#10812](https://github.com/triton-lang/triton/pull/10812)、[#10817](https://github.com/triton-lang/triton/pull/10817) | 2CTA TMA multicast 不属于冻结支持集 |
| merged/product boundary | [#9331](https://github.com/triton-lang/triton/pull/9331) | 至少一个 production kernel 曾因 invalid IR 选择性禁用 WS；适合证明“专用优化”，不证明普适成熟 |

## 5. 历史路线：解释选择，不进入冻结支持矩阵

| 材料 | 历史作用 | 为什么不作为当前支持证据 |
|---|---|---|
| [#5622](https://github.com/triton-lang/triton/pull/5622)、[#5627](https://github.com/triton-lang/triton/pull/5627) | 旧 AutoWS，使用 `num_consumer_groups` / `num_buffers_warp_spec` | 是当前架构前身，不是冻结 pass |
| [#5860](https://github.com/triton-lang/triton/issues/5860) | 旧 heuristic 只覆盖单 producer 与一/两个相同 consumer | 只说明早期限制 |
| [#6689](https://github.com/triton-lang/triton/pull/6689) | ARef 分享分支及受限 partitioner workaround | 未直接合入主线 current pipeline |
| [#6456](https://github.com/triton-lang/triton/pull/6456)、[#6457](https://github.com/triton-lang/triton/pull/6457) | commit 解耦、移除 waiter 的早期尝试 | 后续由 arrive/`is_async` 与 #6514 路线替代 |
| [#7371](https://github.com/triton-lang/triton/pull/7371) | rewrite multiplicity 的独立方案 | 经 #7561 整合后被 #7611 回退，再由 #7645 take-two 重新落地；只作设计路线证据 |
| [#7612](https://github.com/triton-lang/triton/pull/7612) | 早期 semaphore prototype | 完整 redesign 是未合入的 #10121 |
| [#6246](https://github.com/triton-lang/triton/pull/6246)、[#6289](https://github.com/triton-lang/triton/pull/6289) | release/3.3 backport | 不作为冻结 main provenance |
| [#8158](https://github.com/triton-lang/triton/pull/8158) | ARef insertion 曾在调试期临时关闭 | 只说明集成过程经历过主动回退；冻结 pipeline 已重新启用，不能写成当前状态 |

## 6. 明确排除或只旁注的同名噪声

- Hopper CodePartition 主线：#6746、#6624、#6712、#7136、#7658、#7796、#7828、#8260/#9147。只有 legacy `!nvws.token` 消歧需要读取其中的 `WSCodePartition/WSLowerToken`；它们不进入 Blackwell AutomaticWS 支持矩阵。
- AMD/GFX1250 的同名 warp-specialization 工作。
- Gluon explicit WS API；仅在显式/自动模型对照时引用，不能证明 Triton Python AutomaticWS。
- 通用 sanitizer、release backport 与无 WS-specific behavior 的机械重构。
- [#6562](https://github.com/triton-lang/triton/pull/6562) 只修复 NVWS lit tests/CI wiring：它证明测试曾需要接入，不改变当前 operation 或 lowering 语义。
- [#10993](https://github.com/triton-lang/triton/pull/10993) 的 adjacent accumulator RMW：不是当前 WS bug；review 只说明继续累积 pattern detector 会形成设计债务。

## 7. review comments 提供、源码本身不容易讲清的九条因果关系

1. **partition region 是并发 live range。** #5963 指出不能按 MLIR 文本先后建立 register interference；多个 worker regions 在执行期可并发存活。
2. **ARef 表达 ownership，不只是 copy。** #6520/#7581/#8009 同时覆盖 RAW、WAR、storage lifetime 与 owner transfer。
3. **first/last user 是 pipeline time。** #7581 的 review 要求按 stage/cluster 顺序找生命周期边界，block program order 不足以决定 release。
4. **async 是显式语义。** #7590 说明“有无 completion barrier operand”不能代替 `is_async`；TMA 被专门化而 MMA 保持同步是合法组合。
5. **barrier placement 是 frontend correctness。** #6660 指出 PTXAS 不会替前端跨 wait/arrive 调度；#8311 给出错误移动导致 silent wrong result 的实证。
6. **TMEM double buffer 是 graph property。** #8950 要求证明 producer-consumer topology，而不是数 partition changes。
7. **partition attrs 是 pass-local analysis ABI。** #8123/#8534/#8799/#9023/#9133 解释了为何最终选择穷举标注、region arguments 与 explicit holder captures。
8. **register redistribution 与 instrumentation 相互影响。** #9786/#10417 说明 sanitizer/assert/print 的额外 register use 能改变甚至破坏 `setmaxnreg` 协议。
9. **multi-CTA slot 与 phase 必须共同快照。** #10656/#10669/#11010 的 race 是：CTA1 arrive 后停顿，CTA0 完成并翻 phase；CTA1 恢复时若只重读单 bit parity，可能等待下一代 barrier。

## 8. 审计完成标准

一次材料只有满足下面至少一项，才进入正文或成熟度矩阵：

- 能解释冻结源码某个仍存活的结构选择；
- 能给出源码难以直接读出的 correctness/performance trade-off；
- 是 frozen test 所覆盖 shape 的设计 provenance；
- 是 open/draft/fixed failure mode，足以改变支持边界或成熟度措辞。

每项必须再与冻结源码和当前 tests 对账。PR 作者的性能数字、已关闭 bug 的旧 repro、draft 设计都不能替代本地 compile-only 证据。正文最终仍以 `source-ledger.json`、`source-map.json`、`case-matrix.json` 和 `REVIEW_CHECKPOINTS.md` 交叉约束；本文件保存更宽的语料发现与排除过程。
