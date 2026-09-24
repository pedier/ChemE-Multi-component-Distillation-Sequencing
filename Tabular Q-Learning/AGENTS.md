<!--
Last modified time: 2026-09-14-15:04
Last modified content: Require English comments and docstrings throughout the project
Last modified by: OpenAI Codex
File design: Project-level collaboration rules
File purpose: Constrain code structure, chemical semantics, testing, and reporting
File creator: OpenAI Codex
-->

# Agent 指南

本项目实现四组分蒸馏序列问题的 Tabular Q-learning。所有实现必须保持教材塔编号、物理单位、确定性转移和未折扣经济目标一致。

## 实现规则

- 运行时代码仅使用 Python 标准库。
- 外部动作编号固定为 `1..10`，动作掩码索引为 `action_id - 1`。
- 状态顺序固定为 `ABCD, ABC, BCD, AB, BC, CD`。
- 每个 Python 文件不超过500行，每个函数不超过50行。
- 每个文件必须包含分行英文元信息；每个函数和类必须包含英文多行 docstring。
- 超过5行的逻辑代码块前必须添加说明其意图的英文注释。

## 验证流程

1. 运行改动范围的 focused tests。
2. 运行完整 pytest 与 coverage，覆盖率不得低于90%。
3. 运行 mini-linter，并将 warning 视为失败。
4. 任一检查失败时，修复后从完整测试重新开始。
5. 全部通过后，在 `.agents/reports/` 更新阶段报告。
