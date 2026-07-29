# 后续章节素材：从 persistent dispatch 到复杂 case 与成熟度边界

> 冻结范围：Triton `bf64a5db1bc8aab0fd4f0076e60f6c367852e47d`，目标仅为 `sm103`，证据仅限 compile-only。
>
> 本稿只讨论源码、MLIR 变换、LLVM dialect、PTX 结构和编译期测试。它不包含运行时结果，不从 SASS 推导设备行为，也不把仓库中的硬件测试源码当作本机已经执行过的证据。

这部分最适合接在寄存器分配章节之后。前文已经从 `tl.range(..., warp_specialize=True)` 走到 `ttg.warp_specialize`，也解释了 ARef、TMEM ARef 和物理 warp-group 分配。下面回答剩余的五个问题：

1. `ttg.warp_specialize` 最终怎样变成一个长期存活的 worker dispatch loop？
2. scaled/block-scaled MMA 为什么不能只沿用普通 MMA 的“两个矩阵加一个 accumulator”视角？
3. 两个 CTA 共同执行 TCGen05 时，为什么 CTA barrier、cluster barrier 和 WS barrier 必须分层？
4. persistent attention 与 grouped GEMM 分别把自动分区推到了哪些边界？
5. 哪些能力可视为 lowering core，哪些只是受测试约束的 recognized shape？

---

## 13. Case 10：`ttg.warp_specialize` 最终不是一次分支，而是 persistent worker state machine

### 13.1 先建立正确的执行图

降低前，一个 `ttg.warp_specialize` 看起来像一次结构化控制流操作：default region 与若干 partition region 并列存在。降低后，它不再对应“一次 if/else”。同一个 kernel 中所有 worker warps 先进入一个共同的 switch loop，然后由 default warps 在每个 WS site 写入状态，临时派遣 worker 去执行某个 partition：

```text
kernel header
  |
  +-- wid < defaultNumWarps ------> 原 kernel/default 路径
  |
  `-- worker warp ----------------> switchLoop
                                      |
                                      +-- state 0 -> partition A --+
                                      +-- state 1 -> partition B --+--> switchLoop
                                      +-- ... ---------------------+
                                      `-- exit  -> return
```

源码入口是：

- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp`
  - `ConvertWarpSpecializeToLLVM::runOnOperation`
  - `lowerWarpSpecialize`
- `lib/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.cpp`
  - `lowerWarpSpecializeCommon`
  - `rewritePartitionRegions`

`lowerWarpSpecialize` 先收集函数内全部 `WarpSpecializeOp`，读取模块上的 `ttg.total-num-warps`，计算绝对 warp id：

```text
tid = nvvm.read.ptx.sreg.tid.x
wid = tid / threadsPerWarp
isDefault = wid < defaultNumWarps
```

`isDefault` 为真时进入原 kernel entry；否则进入 `switchLoop`。这里有两个容易误读的点。

第一，worker loop 是函数级共享控制骨架，不是每个 WS site 都复制一套永久 loop。`lowerWarpSpecializeCommon` 为所有 `wsOps` 建立同一个 `partitionStates`/`partitionBlocks` 集合，并用 `warpToState[ws-site][relative-worker-warp]` 表示“在这个 site，每个 worker warp 应执行哪个 state”。

第二，状态是按 worker warp 写入 shared memory 的 `i8`，而不是一个 CTA 共享的单值。这样，一个 4-warp partition、一个 2-warp partition以及用于补齐物理 warpgroup 的空闲 warps，可以在同一次派遣中得到不同 state。没有被当前 site 使用的 worker entry 保留 `-1`，写成 `i8` 后落到 switch 的 default destination，在那里只参加协议 barrier，再回到 switch loop。

对应的直接测试是 `test/Conversion/warp_specialize_to_llvm.mlir` 中的：

- `@generate_switch_loop`
- `@multiple_specialize`
- `@cfg`

其中 `@generate_switch_loop` 是阅读 lowering 的首选 fixture；`@multiple_specialize` 证明多个 WS site 共享函数级 dispatch 结构；`@cfg` 证明 partition 内部不必退化成单 basic block。

### 13.2 一次 dispatch 的完整握手

default 路径抵达一个 WS site 时，`lowerWarpSpecializeCommon` 做四件事：

1. 根据 `warpToState`，向 shared state 数组逐项写入每个 worker warp 的 state id；
2. 如果 partition 仍有显式 captures，把它们按 packed LLVM literal struct 写入这个 WS op 的 shared allocation；
3. 执行第一道 all-worker barrier，释放停在 switch loop 的 workers；
4. 调整 default region 的 register budget，再执行第二道 barrier，确保 workers 已读完 captures，之后 shared capture 空间才可被复用。

worker 一侧的镜像动作位于 `rewritePartitionRegions`：

1. partition 开始时按 `actualRegisters` 调整 worker register budget；
2. 每个线程从 packed shared capture struct 中读取自己的值；
3. 执行 barrier，声明 capture 读取阶段结束；
4. 执行 partition body；
5. 把 `ttg.warp_return` 改写为 barrier、恢复低 register budget、跳回 `switchLoop`。

default region 的 `ttg.warp_yield` 也会变成 barrier、register 恢复和到 WS 后继块的 branch。因此，这里有两个不同层次的“完成”：partition 自己的 TMA/TMEM/TCGen05 completion 负责数据依赖；switch-loop barrier 负责 default 与 worker 的控制协议。不能用前者代替后者。

kernel 将要返回时，lowering 会给每个 worker state slot 写入一个额外的 exit state，执行 barrier；switch 中该 state 的目标块包含真正的 `llvm.return`。源码还显式拒绝超过 `uint8_t` 可表达范围的 partition state 数量，诊断文本由 `lowerWarpSpecializeCommon` 产生。这个限制来自状态编码宽度，不是自动分区的 cost model。

### 13.3 capture：能重算就不进 shared，不能重算才传输

capture 处理的源码锚点是：

- `WarpSpecializeUtility.cpp::findTrivialSubcomputation`
- `WarpSpecializeUtility.cpp::elideTrivialCaptures`
- `WarpSpecializeUtility.cpp::rewritePartitionRegions`

`elideTrivialCaptures` 不会机械地把所有外部 SSA 值写到 shared memory。它先反向追踪 capture 的定义：

- kernel entry block argument 可以直接作为重算图的根；
- LLVM 层 `isPure(op)` 的操作可以在每个 partition 中重建；
- 遇到其他 block argument 或非 pure operation 就停止；
- 可重建子图最多 16 个 operations。源码把这个上限明确标成任意的实现阈值，而不是语义限制。

满足条件的子图按拓扑序 clone 到各 partition，capture operand 与 region argument 被删除。多个 partition 产生重复 pure subgraph 时，后续 CSE 再负责清理。不能重算的 capture 才进入 packed shared struct。

对应测试：

- `test/Conversion/warp_specialize_to_llvm.mlir::@pass_captures`
- `test/Conversion/warp_specialize_to_llvm.mlir::@capture_function_arg`
- `test/Conversion/warp_specialize_to_llvm.mlir::@trivial_remat`
- `test/Conversion/warp_specialize_to_llvm.mlir::@remat_subgraph`
- `test/Conversion/warp_specialize_to_llvm.mlir::@no_captures`

教材中应把“capture lowering”拆成两条可分别验证的断言：

- pure 且规模受限的定义链消失，计算出现在 partition 内；
- 未消除 capture 才出现 shared GEP/store/load 与前后 barrier。

不要把“shared capture 变少”写成性能结果；compile-only 能证明的只是 IR 数据通路发生了变化。

### 13.4 partition 内 `ttg.warp_id` 必须重新从零编号

`AllocateWarpGroups` 已经把 worker partitions 放到 default warps 之后，并在 `warpGroupStartIds` 中记录绝对起始 warp id。partition 源程序却应看到局部编号，例如一个从绝对 warp 8 开始、宽度为 4 的 partition，应看到 `0..3` 而不是 `8..11`。

`ConvertWarpSpecializeToLLVM.cpp::rewriteWarpSpecializeWarpIdsOnce` 完成这一步：

```text
relativeWarpId = absoluteWarpId - getWarpGroupStartWarpId(partitionBlock)
```

它保留新的 `ttg.warp_id`，让后续 NVGPU-to-LLVM 正常降低，只在这里插入减法并替换 partition 内旧结果。测试 `test/Conversion/warp_specialize_to_llvm.mlir::@partition_warpid_order` 专门覆盖 partition 排序与相对编号；`@warpid_warp_specialize` 覆盖 WS 内 warp-id 使用。

相对编号不是显示层面的美化。partition 的 layout、lane/warp mapping 与 warp-local 控制都建立在局部 warp-group 坐标上；把物理绝对 id 泄漏进去会改变程序语义。

### 13.5 为什么源码主动禁止 switch loop 的 LLVM LICM

`WarpSpecializeUtility.cpp::disableLICM` 在 switch-loop default block 的回边 branch 上添加：

```text
#llvm.loop_annotation<licm = <disable = true>>
```

源码注释给出的原因很具体：若 LLVM 把 partition 相关代码从生成的 switch loop 外提，会拉长 live range，并可能使 partition region 出现 register spilling。这里应区分两个阶段：

- TTIR/TTGIR 的 LICM 是前端优化与 WS 准备流水线的一部分；
- 这里禁用的是对 lowering 人工构造的 persistent dispatch loop 再做 LLVM LICM。

也就是说，“编译器使用 LICM”与“编译器禁止 LICM”并不矛盾，它们保护的是不同 IR 层次的不变量。`test/Conversion/warp_specialize_to_llvm.mlir::@generate_switch_loop` 和 AMD 对照 fixture 都 FileCheck 了该 loop annotation。

### 13.6 本 case 在 SM103 compile-only 中应保存什么

必选结构证据：

- WS lowering 前的 `ttg.warp_specialize`、`warpGroupStartIds`、`actualRegisters`；
- lowering 后的 header/default-vs-worker branch；
- shared `i8` state 数组、`llvm.switch`、partition blocks、exit block；
- capture rematerialization或 shared packed struct；
- relative warp-id subtraction；
- switch-loop 回边的 LICM-disable annotation；
- `llvm.nvvm.barrier.cta.sync.all` 与 `nvvm.setmaxregister` 的位置。

PTX 可记录 shared load/store、branch、barrier 和 `setmaxnreg` 对应结构；SASS 只作为可选附录，不把具体 opcode 拼写设为通过条件。

一个必须保留的源码限制是：冻结版本的 `ConvertWarpSpecializeToLLVM::runOnOperation` 构造 `NVIDIA::TargetInfo(/*computeCapability=*/100, /*ptxVersion=*/87)`，旁边仍有“假设 WS 只发生在 Blackwell”的 FIXME。顶层编译目标仍是 SM103，但该 helper 的内部 target info 并未从 `sm103` 动态传入。因此，本文可以证明 SM103 编译路径经过这套 lowering，不能把 helper 中的常量解释成精确建模了 SM103 的所有细节。

---

## 14. Case 11：scaled/blockscale 的关键不是多两个 operand，而是分清依赖 token、completion protocol 与 scale ownership

### 14.1 async token 是可选的 TMEM dependency/mod-ref token，不等于硬件 completion

`ttng.tc_gen5_mma`、`ttng.tc_gen5_mma_scaled`、`ttng.tmem_load` 与 `ttng.tmem_store` 的 ODS 定义都把 TMEM dependency token 设计成可选输入/输出。`TCGen5MMA*` 的说明尤其明确：token 在存在时表示 accumulator 上的 TMEM read/write，可供 alias 与 mod/ref 分析使用。它不是“TCGen05 已完成”的硬件事件，也不能单独替代 completion mbarrier。

`lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp::initialDataValues` 对两种 MMA 一视同仁：

```text
TCGen5MMAOp          optional result 0 -> data value（该结果存在时）
TCGen5MMAScaledOp    optional result 0 -> data value（该结果存在时）
TMEMLoadOp           tensor result 0，以及可选 token result 1（存在时）-> data values
```

这条规则的作用是让已经物化的 token use-def 链参与 partition dataflow：MMA 对 accumulator 的 mod/ref 依赖可以沿 TMEM load 与 loop yield 传播。它证明的是编译器依赖建模，不是“token 自身等待了异步硬件”。

直接测试是 `test/TritonGPU/partition-scheduling.mlir::@scaled_mma_with_loads`。它检查：

- descriptor load/local allocation 位于 load partition；
- `ttng.tc_gen5_mma_scaled` 位于 compute partition；
- 此 fixture 显式产生 `%mma_tok`，以它为依赖的 `ttng.tmem_load` 位于 data/consumer partition；
- `scf.yield` 同时携带多个 partition id，并把该 fixture 中的 token loop output 标到正确 producer partition。

硬件 completion 是另一条轴：`is_async=false` 时 op 具有同步语义且不能携带 completion barrier；异步 op 若带 barrier，会对其 commit/arrive，consumer 在 barrier wait 后才观察到 completion。若 producer 与后续 TCGen05 access 跨线程，安全读取还需要 ISA 规定的 ordering；冻结实现缺少 canonical `tcgen05.fence::after_thread_sync`。这些边在图中应分开画：token 是 SSA dependency/mod-ref 边，barrier/commit/wait 是异步完成协议，TCGen05 fence 是专用跨线程排序。

scaled MMA 是否能保持异步还受 scale operand 约束。`LowerAref.cpp::setIsAsync` 只在带多 stage 的 loop 中尝试异步化 ARef consumer；对 scaled op，它同时要求：

- `areScalesPipelineable` 成立，即 loop 内定义的 scale 必须具有 shared encoding；loop 外定义的 scale不受这一项阻止；
- `isOperandPipelineable` 能沿 view/ARef/load 路径证明 A/B scale 可流水化。

任一 scale 条件失败时，该 op 被设为同步。通用 MMAv5 loop pipeliner 的异步路径则显式添加 completion barrier，把 A、B、A-scale、B-scale 都纳入 wait buffer 集合。最终仍为同步语义的 MMAv5 op 由 `MMALowering.cpp::SyncMMALowering` 分配并初始化一个私有 mbarrier，把 op 改写成内部 async form，随后立即 `WaitBarrierOp` 和 `InvalBarrierOp`；这是同步语义的 lowering fallback，不是跨 iteration overlap。因而“生成了 async TCGen05 指令形式”本身也不能证明 scaled MMA 被有效流水化。

### 14.2 scales 有三种来源，不能统一当作普通 SMEM operand

冻结实现至少覆盖三种不同的 scale 路径：

1. scale 已经常驻 shared memory，直接作为 scaled MMA operand；
2. scale 由 tensor descriptor/TMA 载入 shared memory，再 reshape/transpose 成 MMA 需要的布局；
3. scale 被写入 `#ttng.tensor_memory_scales` 编码的 TMEM，再由 compute owner 使用。

第二种路径由 `test/TritonGPU/partition-scheduling.mlir::@scaled_mma_descriptor_scales` 展示。该 fixture 中：

- `tt.descriptor_load` 与最初 `ttg.local_alloc` 在 descriptor/load partition；
- `ttg.memdesc_reshape`、`ttg.memdesc_trans` 以及最终 scaled MMA 在 compute partition；
- accumulator `ttng.tmem_load` 在 consumer partition；
- scaled MMA token 继续跨 loop iteration 作为 iter arg。

第三种路径不能只靠 Shared ARef。`test/NVWS/aref-tmem-insertion.mlir` 中的以下 fixtures 展示 scale 的 TMEM ownership：

- `@matmul_scaled_rhs_scales_tma`
- `@load_scale_mma_user`
- `@nested_loop_yes_double_buffer_scaled`
- `@nested_loop_no_double_buffer_scaled`

以 `@matmul_scaled_rhs_scales_tma` 为例，RHS scale 先由 descriptor load 产生，再写入一个 TMEM-scale ARef buffer；compute partition 通过 `nvws.aref.get.enter`/`nvws.aref.buffer` 获取所有权，执行 `ttng.tc_gen5_mma_scaled`，随后用 `#nvws.async_op<tc5mma>` 标注的 `get.exit` 表达“该 owner 的释放与 TCGen05 异步完成相连”。accumulator 自身还有另一条 TMEM ARef 通道。这说明一个 scaled MMA 周围可能同时存在：

- A/B tile 的 shared producer-consumer通道；
- A/B scale 的 shared 或 TMEM ownership 通道；
- accumulator 的 TMEM ownership/ARef 通道；
- 在 IR 选择物化 token 时，MMA、TMEM load/store 之间的可选 dependency/mod-ref token 链；
- 与 token 分立的 MMA completion barrier、commit/arrive 与 wait。

“多两个 scale operands”远不足以描述它的同步语义。

### 14.3 double buffering 由 accumulator 形状与所有权图共同决定

`@nested_loop_yes_double_buffer_scaled` 检查 accumulator 被改写成首维为 2 的 TMEM buffer；`@nested_loop_no_double_buffer_scaled` 带 `tt.disallow_acc_multi_buffer`，检查首维保持 1。对应决策位于：

- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertTmemAref.cpp`
  - `TmemAccessDag`
  - `TMEMAref`
  - `hasProducerConsumerPartitioning`
  - `insertTmemAref`
  - `workaroundForLoopScheduler`
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp`
  - `multiBufferAref`

这里的“2”是 ownership ping-pong 的 buffer count，不是 `numStages` 的别名。shared/TMA ARef 可以按 `numStages` 展开；TMEM accumulator 是否形成双 buffer，要看访问 DAG、owner 切换和显式禁止标志。冻结实现的精确断言是 `partitions.size() <= 2`：至多两个不同的显式 `(partitionId, wsTag)` owner；未标 partition 的 root owner 由 `hasRootPartition` 另行记录，并可额外计入 `totalOwners`。因此总 owner 数不应误写成“一定最多两个”，但实现同样不支持任意多显式 partition 的 TMEM ownership graph。

### 14.4 从 scaled op 到 block-scale PTX

最终转换位于 `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`：

- `getScaleFactor`
- `getScaleVecSize`
- `isBlock16Scale`
- `getMXFPKind`
- `getScaleKind`
- `createScaleInstDescriptorFp8`
- `createScaleInstDescriptorFp4`
- `createScaledGen5MMA`
- `convertScaledDot`
- `TCGen5MMAScaledOpConversion`

`isBlock16Scale` 用 scale tensor 末维与 block K 的关系识别 block-16；`getMXFPKind` 与 `getScaleKind` 再结合 A/B element type、scale element type、转置条件选择 instruction kind。`createScaleInstDescriptorFp8`/`Fp4` 构造 instruction descriptor，`createScaledGen5MMA` 发出对应 TCGen05 inline PTX。

最终结构测试集中在 `test/Conversion/tritongpu_to_llvm_blackwell.mlir`：

- `@tc_gen5_mma_block_scale`
- `@tc_gen5_mma_block_scale_fp4_a`
- `@tc_gen5_mma_block_scale_nvfp4`
- `@tc_gen5_mma_block_scale_mxfp4`
- `@tc_gen5_mma_scaled_a_tmem`
- `@tc_gen5_mma_scaled_fp8_tmem_lhs`
- `@tc_gen5_mma_scaled_fp4_padded_tmem_lhs`
- `@tc_gen5_mma_scaled_dense_fp4_tmem_lhs`

可要求的 PTX 结构包括：

```text
tcgen05.mma.cta_group::1.kind::...block_scale.block16
tcgen05.mma.cta_group::1.kind::...block_scale.block32
```

以及 scale TMEM address、instruction descriptor 和 predicate operands。具体 descriptor 常量由 shape/type 决定，不宜写成跨 fixture 的统一值。

### 14.5 scaled/blockscale 的支持边界

这条路径在不同证据轴上表现不同：PartitionScheduling、Shared/TMEM ARef 与 MMAv5 conversion 都有 focused tests，target lowering 有多组 PTX FileCheck；frontend tutorial 只证明 `tl.dot_scaled` 表达，本文没有把某个 Python 输入与这些 TTGIR/PTX fixtures 建立生成溯源，也没有 SM103 runtime。不能把它压成一个“成熟度等级”。其限制包括：

- type、scale format、block size、transpose 与 layout 是组合矩阵，不是一个布尔能力；
- `tl.dot_scaled` 教程证明 frontend 表达方式，不自动证明某个组合会形成 autoWS；
- TMEM scale 与 accumulator 都可能引入独立 ownership channel；
- scale pipelineability 决定异步流水化或同步 fallback，不能只看最终 TCGen05 指令名；
- exact SASS opcode 不在 compile-only 必选证据中。

---

## 15. Case 12：2CTA 组件证据束——不要把 `cta_group::2`、cluster hazard 与 WS all-warps lowering 拼成一个未存在的端到端 fixture

冻结树中的 2CTA 证据来自几组各自聚焦的 fixtures，而不是一个从 Python/TTIR 一路贯穿 AutomaticWS、cluster analysis、WS lowering 到 `cta_group::2` PTX 的单一 SM103 case。本节把它们并列，是为了说明这些组件理论上如何衔接；每条结论仍必须归属于实际覆盖它的 fixture。

### 15.1 先分清三层同步

两 CTA case 中至少有三层不同的同步：

| 层次 | 参与者 | 解决的问题 | 代表结构 |
|---|---|---|---|
| WS dispatch | 同一 CTA 的 default 与 worker warps | 派遣、capture 生命周期、worker 返回 | `llvm.nvvm.barrier.cta.sync.all`、shared state switch |
| TCGen05 completion | 发起 MMA 的 warp/CTA 与 accumulator consumer | TCGen05 对 TMEM 的异步写完成；后继 TCGen05 op 另需跨线程 ordering | 冻结实现生成 `tcgen05.commit...mbarrier` 与 wait；ISA canonical `after_thread_sync` fence 在当前 artifact 中缺失 |
| cluster rendezvous | 协作 CTA 及其要求参与的 warps | cross-CTA 可见性与共同进度 | cluster arrive/wait、cluster-scoped mbarrier |

将三者合并成“一个 barrier”会掩盖实际 correctness protocol。WS named barrier 不跨 CTA；TCGen05 commit 只描述特定异步工作完成；cluster barrier 才解决协作 CTA 之间的 rendezvous。

### 15.2 组件 A：两 CTA MMA/TMEM 的直接目标结构

`test/Conversion/tritongpu_to_llvm_blackwell.mlir::@tc_gen5_mma_2ctas` 使用：

```text
"ttg.num-ctas" = 2
"ttng.two-ctas" = true
TMEM encoding: twoCTAs = true
```

它要求转换结果包含：

```text
tcgen05.mma.cta_group::2.kind::f16
tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64
```

`MMAv5.cpp::createGen5MMA` 根据 `getModuleTwoCTAs(op)` 选择 `cta_group::2`；commit lowering 只让 lead CTA 的 warp 0 中 elected thread 发起相应 commit。`test/Conversion/tritongpu_to_llvm_blackwell.mlir::@tmem_copy_2d_2cta` 则覆盖两 CTA TMEM copy。

这些 conversion tests 能证明 target instruction form 被生成；它们没有同时覆盖 AutomaticWS 形成、cluster hazard 插入和 WS all-warps 初始化，更不能证明两个 CTA 在真实设备上成功共驻或该配置值得使用。

### 15.3 组件 B：cluster hazard 在普通 membar 之外单独分析

`lib/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.cpp` 的关键符号是：

- `hasTCGen5CommitCrossCTA`
- `isDistributedMultiCTAOp`
- `requiresCrossCTAMBarrierInitSync`
- `ClusterBarrierAnalysis`
- `runClusterBarrierInsertion`
- `runCrossCTAMBarrierInitSyncInsertion`

`isDistributedMultiCTAOp` 不只识别两 CTA MMA。它还识别跨 CTA layout conversion/reduction、multi-CTA TMEM copy、multicast TMA、multicast/from-CTA barrier 操作等。分析寻找 unresolved cross-cluster dependency，在需要的位置插入 cluster barrier；`requiresCrossCTAMBarrierInitSync` 还检查 mbarrier allocation 本身是否跨 CTA，或者一个看似 per-CTA 的 barrier 是否会被 multi-CTA consumer 广播使用。

顶层顺序位于 `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp::ConvertTritonGPUToLLVM::runOnOperation`：

1. 建立 `ModuleAllocation`；
2. `runClusterBarrierInsertion`；
3. `runCrossCTAMBarrierInitSyncInsertion`；
4. `ModuleMembarAnalysis::run`；
5. `runClusterBarrierMbarAllocator`；
6. 再进行 dialect conversion。

这个顺序说明 cluster barrier 不是 LLVM/PTX 末端临时补丁。它依赖 allocation alias 与跨 CTA dataflow 分析，必须在相关高层信息消失前决定位置。

### 15.4 组件 C：WS 内 cluster barrier 的双 mbarrier-slot allocator

分配器位于：

- `include/triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierMbarAllocator.h`
- `lib/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierMbarAllocator.cpp::runClusterBarrierMbarAllocator`

冻结常量为：

```text
kClusterBarrierMbarSlotSize       = 16 bytes
kClusterBarrierMbarBufferCount    = 2
kClusterBarrierMbarAllocationSize = 32 bytes
```

只有位于 `ttg.warp_specialize` 内且 `needsClusterBarrier(op)` 为真的 region 才获得 `ttg.mbar_offset`。每个需要独立协议的 WS region 占 32 bytes；模块记录 `ttg.ws_cluster_barrier_count`，并扩展 `ttg.shared`。

测试 `test/TritonNvidiaGPU/cluster-barrier-mbar-allocator.mlir::@cluster_barrier_mbar_allocator` 很适合解释 region granularity：default region 内多个需要 cluster barrier 的 ops 复用同一 32-byte allocation，worker partition 的 cluster barrier 得到另一 allocation。该 fixture 的 `ttg.shared` 从非对齐初值先对齐，再增长为两个 32-byte 区域。

两个 slots 是为防止迟到 CTA 错过 phase。`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ClusterOpsToLLVM.cpp::ClusterBarrierOpConversion` 使用一个模 4 counter：

```text
slot   = counter & 1
parity = counter >> 1
next   = (counter + 1) & 3
```

同一个 slot 因而要隔一次 rendezvous 才复用。单 slot 加 parity 在某 CTA 严重滞后时不足以阻止 peer 连续复用同一个 mbarrier；双 slot 把复用间隔拉开一个 rendezvous。

### 15.5 组件 D：WS all-warps cluster 初始化为何发生在最终 WS lowering 之前

顶层 `make_llir` 中的相关顺序是：

```text
add_to_llvmir
add_initialize_ws_cluster_barriers
canonicalize LLVM IR
CSE
add_warp_specialize_to_llvm
```

`ClusterOpsToLLVM.cpp::InitializeWSClusterBarriers::runOnOperation` 在 LLVM function entry：

1. 由 thread 0 初始化每个 allocation 的两个 mbarrier，arrival count 为 `numCTAs - 1`；
2. 把 counter 写为 0；
3. 发出 mbarrier-init release cluster fence；
4. 让所有 default 与 worker warps 执行 cluster arrive/wait。

第四步通过 `lowerClusterSyncForAllWarps` 实现。如果 kernel 有 worker warps，该 helper 临时构造一个覆盖全部 worker 的 `ttg.warp_specialize`：default region 与每个 worker partition 都发出相同 cluster sync。随后正常 WS lowering 再把它折成 persistent dispatch protocol。这样不会只同步 default warps 而把常驻 switch loop 中的 workers 漏掉。

对应测试集中在 `test/Conversion/tritonnvidiagpu_to_llvm.mlir`：

- `@cluster_barrier_inside_warp_specialize`
- `@relaxed_cluster_barrier_inside_warp_specialize`
- `@cluster_barrier_inside_warp_specialize_reuses_slots`
- `@cluster_arrive_warp_specialized`
- `@cluster_wait_warp_specialized`
- `@cluster_barrier_warp_specialized`

`@cluster_barrier_inside_warp_specialize` 检查两个 mbarrier init、counter load、slot/parity 计算、peer arrive、parity wait、counter 更新与 CTA-local 前后 barrier；`@...reuses_slots` 检查同一 region 的多个 cluster barriers 复用 allocation，而不同 WS regions 获得独立 allocation。

### 15.6 SM103 compile-only 的结论边界

这组 focused evidence 可以分别确认：

| 证据组件 | 能确认 | 不能据此确认 |
|---|---|---|
| `tc_gen5_mma_2ctas` / `tmem_copy_2d_2cta` | conversion 可选择 `cta_group::2` TCGen05/TMEM form | 同一输入也经历了 autoWS 和 cluster hazard 插入 |
| ClusterBarrierAnalysis/init-sync fixtures | 特定 cross-CTA dataflow 会触发 cluster barrier 与初始化同步 | 该 dataflow 最终一定来自上述 2CTA MMA fixture |
| cluster mbar allocator fixtures | 需要协议的 WS region 获得两槽 allocation 与 offset/count metadata | 两个真实 CTA 已成功 launch/共驻 |
| WS cluster lowering fixtures | 初始化与 rendezvous 可覆盖 default/worker warps，并生成 slot/parity protocol | 与 `cta_group::2` MMA 在一个端到端编译产物中共同出现 |
| Gluon multi-CTA tutorial | 人工 multi-CTA/WS 程序的可读高层形状 | Triton AutomaticWS 对同一 Python shape 的端到端资格 |

这些组件的组合仍不能确认：

- cluster launch 在当前机器上的设备合法性；
- 两 CTA 是否能按预期共驻；
- barrier 频率是否合理；
- 一 CTA与两 CTA的任何性能关系。

因此这里应记录一个证据向量：`cta_group::2` target conversion、cluster analysis、allocator 与 WS all-warps lowering各自有 focused coverage；单一 composed SM103 fixture、设备 launch/residency、数值与性能轴为空。不能再概括成“2CTA compiler path 已端到端完整”。

---

## 16. Case 13：persistent attention——自动分区必须穿过两层 loop 和两套 accumulator 生命周期

### 16.1 高层 shape 为什么比 GEMM 难

Python integration source 是 `python/test/unit/language/test_warp_specialization.py::attention_persistent_inner_loop_kernel`。它只有外层 tile-claim loop 带：

```python
for _ in tl.range(0, tiles_per_sm,
                  warp_specialize=warp_specialize,
                  num_stages=num_stages):
```

内层 `start_n` loop 自身没有再次写 `warp_specialize=True`，但它包含：

1. K descriptor load；
2. QK `tl.dot`，结果进入第一块 TMEM accumulator；
3. TMEM load 后的 max/sum/exp2 等 softmax register work；
4. V descriptor load；
5. P×V `tl.dot`，写第二块 TMEM accumulator；
6. 该 fixture 显式物化并跨 iteration 携带的 `m_i`、`l_i` 与两个 TMEM dependency tokens。

因此 PartitionScheduling 不能只在带 marker 的 loop body 顶层找几条相邻 operations。它必须递归构建内层 loop graph，让 inner loop 的 descriptor/TMEM/MMA data roots 向外层 marked loop 传播，同时保留 outer persistent tile index 的独立更新。

### 16.2 focused partition fixture 给出的四类角色

最精确的 pass 级编译证据是独立的 TTGIR fixture `test/TritonGPU/partition-scheduling.mlir::@attention_persistent_inner_loop_kernel`。它检查内层 loop 的 outputs 最终带：

```text
ttg.partition = array<i32: 0, 1, 2, 3>
ttg.partition.outputs = [
  array<i32: 0>,
  array<i32: 0>,
  array<i32: 2>,
  array<i32: 1>
]
```

不要把编号解释成永久 ABI；它们只是冻结 pass 的 serialization 结果。但这个 fixture 可以稳定说明存在四类 dataflow 角色：

- QK TMEM load 与 softmax register work 在一个 consumer/data partition；
- accumulator rescale、TMEM store 与 P×V MMA 形成另一条 accumulator owner chain；
- descriptor/TMA producer work形成 load role；
- outer persistent `tile_idx += num_sm` 具有独立的标量循环状态，测试把该 add 标入第四 partition。

该 fixture 内层 loop 的两个可选 TMEM dependency tokens 已被显式物化，并分别映射回其 producer partitions；`m_i`/`l_i` register outputs 则留在 softmax partition。这里正好展示 `PartitionLoops.cpp::classifyLoopVars` 为什么必须区分：forwarded、computed、captured 与 partition output，而不能把整个 iter_args 元组绑定到一个 partition。

同名并不构成生成溯源。冻结快照没有在这两份材料之间记录“该 TTGIR 正由上述 Python kernel 在同一 commit、同一 options 下生成”的 hash/命令链；因此 Python 文件证明 integration input 与参数边界，TTGIR 文件证明 PartitionScheduling 的局部契约，本文不把二者伪装成一次连续编译的前后快照。

### 16.3 ARef 与 TMEM ARef 在 attention 中各管哪一段

相关 focused tests 是：

- `test/TritonGPU/partition-scheduling.mlir::@attention_forward`
- `test/NVWS/aref-tmem-insertion.mlir::@attention_forward`
- `test/NVWS/lower_aref.mlir::@attention_forward`
- `test/TritonGPU/automatic-warp-specialization.mlir::@attention_forward`

这些非 persistent attention fixtures 用较小控制结构隔离完整后续 passes，persistent fixture 则重点覆盖 nested partition scheduling。二者结合可用来解释：

- K/V descriptor load 到 shared tile consumer：Shared ARef + TMA mbarrier stage protocol；
- QK MMA 到 softmax TMEM load：TMEM ARef owner hand-off；
- softmax 对旧 accumulator 的 rescale，再到 P×V MMA：同一 TMEM buffer 上 store/MMA/token 串联；
- `m_i`、`l_i`、row max 等普通 tensor/register values：partition output 或可复制计算，而不是 TMEM owner token。

不要声称 persistent fixture 已在每个独立 NVWS pass 文件中都有同名 test。冻结测试策略是：persistent Python source 与 persistent PartitionScheduling TTGIR 是两份独立证据；较小 attention TTGIR shape覆盖 InsertTmemAref、LowerAref 和完整 AutomaticWS。三层证据可以互相解释机制，但不能拼接成已验证的单一 provenance chain。

### 16.4 compile-only 学习 case 的建议切片

后续补实验时，这个 case 不应直接从 Python 一步跳到 PTX，否则四类角色会被最终 CFG 淹没。更适合由同一次编译保存五个带命令、options 与 hash 的切片：

1. TTIR：只有 outer loop 带 `tt.warp_specialize`，inner loop 包含 QK/softmax/PV；
2. PartitionScheduling 后：观察 outer/inner loop 的 `ttg.partition.outputs`；
3. InsertAref/InsertTmemAref 后：分别数 shared channel 与 TMEM owner channel；
4. LowerAref/PartitionLoops/LowerWarpGroup 后：观察 default/worker regions 与 loop arguments；
5. 最终 LLVM/PTX：只验证 persistent dispatch、TMA、TCGen05、TMEM 和 barrier 结构共存。

Python 文件还定义了 `test_warp_specialize_attention_persistent_forward`，并包含多组 shape 与显式 shared-memory skip 条件。本文可以引用它作为仓库的 integration fixture 与已知配置边界，但在没有 SM103 执行时，不能转述其数值比较结果。特别是 `BLOCK_M=128`、`HEAD_DIM=128`、非 FP8 的部分 warp/stage 组合会因为 shared memory 需求被跳过，这本身说明支持不是“所有参数组合均可编译”的单一布尔值。

### 16.5 证据向量

persistent attention 具有真实 Python integration source、独立 persistent PartitionScheduling fixture，以及较小 attention shape 的 ARef/TMEM-ARef/AutomaticWS tests；但同一输入的 composed pass artifact 与 SM103 runtime 仍为空。其自动策略/泛化轴还有以下限制：

- exact partition membership 依赖 heuristic merge；
- nested loop outputs 与两个 accumulator ownership chains 增大变换敏感性；
- shared-memory 配置存在明确边界；
- 本文没有 SM103 数值或 runtime 证据。

因此应逐轴保留这些事实，不给它分配一个会掩盖 provenance 缺口的总等级。

---

## 17. Case 14：grouped GEMM——descriptor 在 persistent loop 内动态重建

### 17.1 它测试的不是另一种矩阵形状，而是 descriptor lifetime

源码 fixture 是：

- `python/test/unit/language/test_warp_specialization.py::grouped_matmul_tma_kernel`
- 同文件 `group_gemm_tma_fn`
- 同文件 `test_grouped_gemm`

kernel 的外层 group loop 带 `warp_specialize=True`。每次 group iteration 都从指针数组和 leading-dimension 数组读取新的 A/B/C 基址与 stride，然后在 loop 内调用 `tl.make_tensor_descriptor`；接着是跨 SM 分发的 tile loop和 K-reduction loop。

与固定 GEMM 相比，核心问题变为：旧 iteration 的 TMA 仍可能引用某份 descriptor storage 时，下一 group iteration 能否安全写入新的 descriptor？普通 SWP 只 multibuffer tile data 并不自动解决 descriptor update lifetime。

### 17.2 `numStages + 1` descriptor buffers 来自明确的重叠窗口

实现位于 `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp::multiBufferTMADescriptors`。它遍历带 WS marker 的 loops；如果其中的 `MakeTensorDescOp` 位于某个 `scf::ForOp`，就把该 enclosing loop 加入 `descUpdateLoops`，随后调用 `lowerTMADescriptors`。

buffer 数不是 `numStages`，而是：

```text
numDescs = numStages + 1
CoarseSchedule(numDescs + 1)
```

源码注释解释了第一个 `+1`：下一次 descriptor update 可以与最老的 in-flight TMA load 重叠，所以必须多留一份 descriptor，避免覆盖仍被引用的 storage。第二个 `+1` 是 `CoarseSchedule` 的编号约定：其 `numStages` 表示最大 pipeline stage 加一；要得到 `n` 个 buffers，需要传 `n+1`。

这段处理放在 `LowerWarpGroup` 与第二次 `ScheduleLoops` 之后、`clearInternalWarpSpecializationAttrs` 之前。它不能完全依赖通用 SWP，因为源码明确要求支持 nested loops 中的 descriptor updates。

### 17.3 完整 AutomaticWS fixture 证明了哪些结构

`test/TritonGPU/automatic-warp-specialization.mlir::@grouped_matmul_tma_kernel` 是本 case 的主要 TTGIR compile-only 证据。其检查包括：

- outer group loop 中的 `ttng.tensormap_create`；
- global scratch allocations，用于动态 tensor map storage；
- default、`partition0`、`partition1`；
- inner tile/K loops 仍然存在；
- TMEM accumulator 被 multibuffer 成首维为 2 的 allocation；
- pass 结束后不残留 `ttg.partition` 与 `ttg.warp_specialize.tag`。

文件头的三条 RUN pipeline 分别覆盖：

1. AutomaticWS 基本结果；
2. 再运行 `-tritongpu-pipeline`；
3. 再运行 `-tritongpu-optimize-partition-warps`。

因此，grouped GEMM 并非只有 Python 源码存在；冻结树中还有一份以 post-frontend TTGIR 为输入的完整 AutomaticWS focused fixture。但仓库没有记录它就是由 17.1 的 Python kernel 在冻结 options 下生成的可验证 provenance；两者必须作为 `Python integration source` 与 `TTGIR full-pass fixture` 两个独立证据槽。

### 17.4 为什么 outer loop 可以是 default 工作，inner compute 仍进入 workers

group pointer load、descriptor creation、tile-index arithmetic与 TMA/MMA dataflow属于不同角色。PartitionScheduling 可以让 descriptor/TMA producer 与 TCGen05/TMEM compute成为 partitions，同时保留控制外壳和某些动态 descriptor 更新在 default region。`PartitionLoops` 随后不是把外层 loop 整体搬进某一个 worker，而是按 partition assignment clone必要的 nested control structure、captures 和 outputs。

这正是 `PartitionLoops.cpp` 中以下符号在真实 workload 上的用途：

- `cloneForOp`
- `cloneIfOp`
- `cloneOpsInBlock`
- `classifyLoopVars`
- `triton::gpu::partitionLoop`

grouped GEMM 也说明“default region 等于 epilogue”是错误简化。default 是未被 worker partition 接管的控制与计算路径；在动态 descriptor case 中，它可以持有影响后续 TMA producer 的控制/descriptor lifetime work。

### 17.5 SM103 compile-only 应检查什么

必选：

- Python→TTIR 后 outer group loop 的 `tt.warp_specialize`；
- `ttng.tensormap_create` 与 dynamic descriptor scratch；
- post-autoWS 的 default/worker partition 结构；
- descriptor buffer 数与 `numStages + 1` 关系；
- internal partition attrs 已清除；
- TCGen05 MMA、TMEM accumulator、TMA load/store 与 persistent LLVM switch共存。

可选 PTX：

- dynamic address/control flow；
- TMA tensor bulk copy与 mbarrier；
- TCGen05 MMA/commit；
- WS state/barrier CFG。

不能从这些结构推出 group 间负载均衡、descriptor update 成本或 cache behavior。

### 17.6 证据向量

grouped GEMM 的 Python integration source 与完整 AutomaticWS TTGIR fixture 两个轴都有证据，后者还覆盖 pipeline/warp optimization 三条 RUN；但两者之间没有冻结 provenance，同一输入的 SM103 target artifact 与 runtime 轴也尚未补齐。它依赖动态 tensormap scratch、nested descriptor lowering 与特定循环组织，因而泛化范围不能外推到任意 ragged/grouped kernel。

---

## 18. 支持证据向量：把“代码存在”“局部变换”“组合编译”“运行有效”作为独立坐标

### 18.1 本文采用的独立证据轴

| 证据轴 | 记录什么 | 该轴不能替代什么 |
|---|---|---|
| Frontend/API | Python kernel、公开 hint 或显式 IR 输入是否存在 | 不证明 frontend 输入会命中 autoWS |
| Automatic discovery/policy | eligibility、PartitionScheduling 与 heuristic 是否对该 shape 有直接 fixture | 不证明下游 target lowering 或泛化到相邻 shape |
| Focused transform | ARef、TMEM ARef、loop partition、allocator、cluster 等局部 pass 契约 | 不证明这些局部 fixtures 来自同一个输入 |
| Composed/full-pass | AutomaticWS 或多 pass RUN 是否在一份冻结 TTGIR 上贯通 | 不自动连接到同名 Python source，除非保存 provenance |
| Target lowering/artifact | LLVM/PTX FileCheck，或本地 SM103 IR/PTX/cubin artifact | 不证明设备可启动、数值正确或存在 overlap |
| Runtime qualification | 同 target 的数值、launch、profiling、重复统计 | 不能由任何 compile-only 轴推断 |
| Generality boundary | dtype/layout/shape/owner 数、FIXME 与 negative cases | 不能被其他轴的“强证据”抵消 |

这些轴没有总分、先后等级或“取最低层”的规则。例如 scaled PTX conversion 有直接证据，与任意 `tl.dot_scaled` frontend shape 能否自动分区是两个独立问题；persistent attention 的 Python source 也与 SM103 runtime 轴彼此独立。

### 18.2 case 证据向量

| Case | Frontend/API | Automatic discovery | Focused transform | Composed/target lowering | Runtime | Generality boundary |
|---|---|---|---|---|---|---|
| 显式 WS micro IR | 显式 `ttg.warp_specialize` + Python compile fixture | 不适用 | switch/capture/warp-id/CFG fixtures | final LLVM lowering focused coverage | 无 | 绕过自动分区，只证明执行协议 |
| 显式 TMEM WS | 显式 TMEM/WS fixture | 不适用 | TMEM allocation fixture | TMEM 与 WS conversion 分别覆盖 | 无 | 不证明 TMEM ARef 自动发现 |
| 只有普通 async copy | marker 存在 | negative：不命中 eligible descriptor memory | no-partition/no-eligible tests | 明确检查无 `ttg.warp_specialize` | 无 | snapshot-specific negative contract |
| TMA + 普通 async mixed GEMM | 参数化 Python source | 命中 | mixed-load AutomaticWS fixture | AutomaticWS + pipeline + warp optimization | 无 | 至少一个 TMA；不代表 pointer-only autoWS |
| Canonical TMA autoWS GEMM | Python fixture | 中心 recognized shape | Partition/ARef/TMEM-ARef/LowerAref 多层 | AutomaticWS 与 target checks | 无 | frontend 文档仍聚焦 simple matmul loop |
| Persistent TCGen05 GEMM | Python source | nested partition fixture | HoistTmemStore 窄 pattern | 分层集成源码/fixture，需保存同次编译 provenance | 无 | nested/hoist 条件敏感 |
| Shared ARef | 间接 | autoWS 中间机制 | 多 consumer/conditional/nested focused tests | 被若干 full-pass fixtures 消费 | 无 | pass-private IR，不是 ABI |
| TMEM ARef | 间接 | autoWS 中间机制 | GEMM/attention/scaled/nested focused tests | 被下游 conversion 消费 | 无 | 至多 2 个显式 partition IDs，另可有 root owner |
| Worker warp/register policy | 间接 | WS 形成后运行 | warp shrink、allocation、dynamic-register tests | `nvvm.setmaxregister` lowering | 无 | heuristic/requested regs 不等于 occupancy |
| Scaled/blockscale | tutorial 表达 | selected TTGIR shapes | partition + Shared/TMEM ARef fixtures | 多种 block-scale PTX FileCheck | 无 | token 可选；async 受 scale pipelineability；组合矩阵敏感 |
| 2CTA component bundle | Gluon manual tutorial | 未建立单一 autoWS 输入 | cluster analysis/allocator/all-warps 分别覆盖 | `cta_group::2` 与 cluster PTX 分散在不同 fixtures | 无 | 无单一端到端 SM103 fixture；无 launch/residency |
| Persistent attention | Python source独立存在 | persistent PartitionScheduling fixture | 较小 attention 的 ARef/TMEM-ARef/AutomaticWS | 未建立 Python→同名 TTGIR provenance | 无 | 多 owner/nested outputs/shared-memory 边界 |
| Grouped GEMM | Python source独立存在 | post-frontend TTGIR fixture | AutomaticWS/pipeline/warp optimization 三条 RUN | TTGIR full-pass；未建立与 Python 的 provenance | 无 | dynamic descriptor shape，不推广到任意 ragged kernel |

### 18.3 机制证据向量

| 实现机制 | 默认 SM103 pipeline | Focused/negative evidence | Target-lowering evidence | Generality/runtime gap |
|---|---:|---|---|---|
| frontend marker 与 Blackwell gate | 是 | no-eligible-memory no-op | 不适用 | hint 而非形成 partition 的承诺；无 runtime |
| PartitionScheduling | 是 | graph/merge/no-root/attention/scaled/persistent | 不直接适用 | partition policy 为 heuristic，编号非 ABI |
| PartitionLoops | 是 | unresolved SSA 负例 | 不直接适用 | 依赖前序 ownership rewrite |
| Shared ARef/LowerAref | 是 | 多 consumer、conditional、nested | mbarrier/TMA 下游覆盖 | 同步结构不等于最佳 overlap |
| TMEM ARef | 是 | 非 WS、多 use、double-buffer on/off | TCGen05/TMEM 下游覆盖 | <=2 explicit partition IDs + optional root owner；非任意 graph |
| OptimizePartitionWarps | 是 | TMA/TMEM minimum、register heuristics | setmaxnreg 下游覆盖 | 不是 occupancy model |
| AllocateWarpGroups | 是 | padding、assert/print、instrumentation | setmaxnreg 下游覆盖 | 仍有 TMEM 对齐 FIXME |
| persistent LLVM switch | 是 | captures、CFG、multi-site、remat、relative wid | barrier/branch/setmaxnreg | 没有运行 overlap 证据 |
| scaled/blockscale conversion | 条件触发 | type/layout、scale pipelineability fixtures | TCGen05 block-scale variants | 同步 fallback 与异步流水化必须分辨；无完整组合矩阵 |
| 2CTA components | `num_ctas=2` 条件路径 | cross-CTA init、slot allocator、WS all-warps 分别测试 | cluster mbarrier 与 `cta_group::2` 分别测试 | 没有证明这些组件同处一个 end-to-end artifact；无运行资格 |

### 18.4 写入正文时应坚持的措辞

可以写：

- “冻结源码在 SM103 编译分支中默认运行 AutomaticWarpSpecialization。”
- “该 fixture 经过 focused FileCheck，期望生成某个 partition/ARef/LLVM/PTX 结构。”
- “仓库包含对应 Python integration kernel；本文只把它作为输入 shape 与支持意图的证据。”
- “SASS 若能生成，仅用于与 PTX/LLVM 控制流对照。”

不要写：

- “compile-only 结构已经证明运行时收益。”
- “worker warps 提高了 occupancy。”
- “测试存在，所以所有参数组合均受支持。”
- “看到 `USETMAXREG` 或 TCGen05 opcode 就证明 overlap 已发生。”

### 18.5 本文的证据截止线

数值执行、设备 launch、profiler counters 与任何运行时比较都属于另一份待补实验，不进入本稿。本文可以确认默认 SM103 pipeline 包含从 frontend hint、dataflow partition、ownership protocol、physical warp-group allocation 到 persistent LLVM/PTX state machine 的实现；但只有具体 fixture 覆盖的边界才算证据。target lowering、automatic-policy generality、Python→TTGIR provenance、runtime qualification 等轴必须分别报告，不能合成为一个“整体成熟度”。
