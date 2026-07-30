# 参考资料研究路由（已收口）

这份文件原先是正文参考资料章的中间稿。为避免同一批 PR、issue 和源码索引在两处复制后继续漂移，冻结提交 `bf64a5db1bc8aab0fd4f0076e60f6c367852e47d` 的最终资料按下面四层维护：

1. 人类阅读的完整参考索引：`../article.md` 第 21 章。每组都先列冻结源码与测试，再列一手硬件文档、论文、issue、PR 或官方 roadmap，并说明它在正文中回答哪个“为什么”。
2. GitHub 广域发现与排除过程：`github-discussion-audit.md`。它记录 title/body/comments/reviews、本地 git history 与 blame provenance 的查询口径、历史分支、open/draft 边界和同名噪声。
3. 可机读证据账本：`source-ledger.json`。其中 `sources` 是逐项炼化的核心材料，`github_discussion_audit.indexed_github_urls` 是更宽的发现索引；二者不能互换。
4. 结论到源码、symbol、fixture 和 case 的映射：`source-map.json`、`case-matrix.json` 与 `../REVIEW_CHECKPOINTS.md`。

收口原则：外部讨论只用于解释设计动机、失败模式与维护者判断；冻结树决定当前 pass 顺序、支持形状和 fail-safe 行为。已关闭 bug 不写成当前仍复现，draft proposal 不写成已实现，PR 的性能数字不替代本项目缺失的 SM103 runtime 证据。

“完整审计”在这里指截至 2026-07-30、按已记录查询式可复现的系统性覆盖，不声称数学上枚举了已删除/编辑内容、未索引 branch、Slack、邮件或内部讨论。这个边界同时写入正文、广域审计和 JSON 账本，另一台机器拉取本分支后不需要依赖本机浏览缓存。
