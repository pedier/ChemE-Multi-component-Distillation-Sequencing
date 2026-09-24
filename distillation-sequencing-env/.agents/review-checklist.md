<!--
Last modified time: 2026-09-14-15:04
Last modified content: Translate all metadata comments into English
Last modified by: OpenAI Codex
File design: Stage review checklist
File purpose: Prevent omissions in state, cost, testing, and documentation requirements
File creator: OpenAI Codex
-->

# Review Checklist

- [ ] 教材塔编号、分离任务和参数保持一致。
- [ ] 奖励是未折扣年化成本的负值。
- [ ] 非法动作不会进入动作集合或状态转移。
- [ ] 状态和转移保持 Markov 性与确定性。
- [ ] 每个函数有对应测试，整体覆盖率至少90%。
- [ ] 文件、函数、元信息、docstring 和代码块注释符合规范。
- [ ] pytest 和 mini-linter 均通过。
- [ ] `.agents/reports/` 已记录最终结果。
