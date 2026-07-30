# Triton Blackwell Warp Specialization 源码课

这是一份锁定到 Triton `bf64a5db1bc8aab0fd4f0076e60f6c367852e47d`、LLVM
`850a2b1b975c061ae0fc982ba68064d305485cb2` 的 SM103 源码解读与编译态实验。
核心交付物是 [article.md](article.md) 和单文件离线版 [index.html](index.html)。

项目没有用本机 SM120 结果冒充 SM103 runtime 证据：冻结源码在 SM120
选择 consumer-Blackwell MMA v2，并排除本文中心的 MMAv5/TCGen05 与 cluster
路径。正文给出这一取舍的源码依据；实验因此固定为 SM103 compile-only。

本文不介绍 CUDA、MLIR 或矩阵乘基础。它从 `tl.range(...,
warp_specialize=True)` 出发，逐层解释：

- 自动 WS 何时有资格发生，何时只是一个未兑现的请求；
- TCGen05 alloc/MMA/commit/fence/ld-st-wait 的因果链，以及本文涉及的
  proxy、tensormap、mbarrier-init 与 TCGen05 fences 为什么互不替代；
- NVWS dialect 的全部类型、属性、接口和 16 个 op，及其 ARef/warp-group 两个
  transient lowering epoch；
- `PartitionScheduling` 如何建立并合并 dataflow partitions；
- shared ARef、TMEM ARef 如何把跨 partition SSA edge 变成所有权协议；
- loop cloning、pipeline、warp-group allocation 和动态寄存器预算如何衔接；
- `ttg.warp_specialize` 如何降成持久 worker switch-loop；
- canonical GEMM、mixed load、persistent GEMM/attention、grouped GEMM、scaled MMA
  与 2CTA case 各自能证明什么，以及当前实现的限制在哪里。

## 建议阅读路径

1. 先读正文第 1–5 章，建立指令因果、NVWS 瞬时 IR、同步分类和显式 WS 语义。
2. 第 6–12 章跟随 canonical TMA GEMM 走完 AutomaticWS 的全部 subpasses。
3. 第 13–14 章把逻辑 partition 落到物理 warp、寄存器和 LLVM 控制流。
4. 最后用边界 cases 和成熟度矩阵判断真实 workload 是否落在当前支持中心。
5. 按 [lab/README.md](lab/README.md) 重建 SM103 编译证据；待有合适设备后再按 [lab/TEST_PLAN.md](lab/TEST_PLAN.md) 的未执行清单补运行资格化，不能从当前结果推导性能。

## 目录

```text
blackwell-warp-specialization/
├── article.md                 # 中文教材正文
├── index.html                 # 自包含、可搜索的离线 HTML
├── assets/                    # 原始插图与 HTML 样式/交互源
├── lab/                       # 不加载、不启动 cubin 的 SM103 编译实验
├── evidence/                  # 经校验后才允许发布的精选编译证据
├── research/                  # 来源账本、源码地图、case 矩阵、GitHub 讨论审计与检索记录
└── tools/                     # 离线 HTML 构建器
```

`research/github-discussion-audit.md` 记录 GitHub issue/PR/comment/review 的检索面、
纳入/排除规则和无法绝对全量化的边界；其余 `research/` 文件保存 source ledger、
machine-readable source map 和 case matrix。它们不是正文阅读前置条件。正文只在最后一章集中列外部资料，
源码路径和符号则就近出现，方便在当前 checkout 中直接跳转。

## 构建离线 HTML

```bash
python3 -m venv .book-venv
.book-venv/bin/pip install -r docs/blackwell-warp-specialization/tools/requirements.txt
.book-venv/bin/python docs/blackwell-warp-specialization/tools/build_book.py
.book-venv/bin/python docs/blackwell-warp-specialization/tools/build_book.py --check
```

生成文件内嵌 CSS、JavaScript 和 SVG；除点击参考资料链接外，阅读不依赖网络。

## 证据边界

所有实验都显式使用 `GPUTarget("cuda", 103, 32)`。允许 `ptxas` 把 PTX 组装为
cubin，也允许把 cubin 作为数据反汇编，但实验工具没有 kernel launch、driver context
或 cubin loader 路径。因此本项目能够证明 pass 发生、IR 结构形成、PTX/SASS 被工具链接受；
不能证明数值正确、芯片上无死锁、实际并发、occupancy 或加速比。

本次审计还发现一个需要单独保留的资格缺口：冻结 lowering 和 exact-SM103
PTX/SASS 都没有 ISA canonical MMA→TMEM-load 路径中的
`tcgen05.fence::after_thread_sync`。因此现有 4/4 PASS 只代表 manifest 声明的结构
合同通过，不代表跨 role TCGen05 ordering 或 runtime correctness 已被证明。
