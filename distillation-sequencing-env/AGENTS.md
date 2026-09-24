<!--
Last modified time: 2026-09-24
Last modified content: Define shared chemical semantics and quality checks
Last modified by: OpenAI Codex
File design: Project collaboration rules
File purpose: Keep all four learners on one verified chemical model
File creator: OpenAI Codex
-->

# Agent guide

This package owns the fixed four-component distillation environment shared by Tabular Q-learning, DQN, REINFORCE, and PPO. Preserve textbook action IDs `1..10`, state order `ABCD, ABC, BCD, AB, BC, CD`, deterministic sharp splits, and negative annual-cost rewards in M$/yr.

- Runtime code uses only the Python standard library.
- Each Python file has separate English metadata lines. Each function and class has a multiline English docstring listing inputs and returns.
- Add an English intent comment before logical blocks longer than five lines.
- Keep Python files at or below 500 lines and functions at or below 50 lines.
- Run focused tests, full pytest with at least 90% package coverage, and mini-linter with zero warnings after changes. If a gate fails, fix it and rerun from the full test gate.
- Update a stage report in `.agents/reports/` after all checks pass.
