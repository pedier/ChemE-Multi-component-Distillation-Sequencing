<!--
Last modified time: 2026-09-24
Last modified content: Allow the dedicated shared chemistry package while preserving learner isolation
Last modified by: OpenAI Codex
File design: Project-level collaboration contract
File purpose: Preserve textbook semantics and user-requested delivery gates
File creator: OpenAI Codex
-->

# Agent guide

This independent project implements masked PPO for the fixed four-component distillation sequence in Example 17.3.

- Preserve textbook tower IDs `1..10`, mask index `action_id - 1`, and state order `ABCD, ABC, BCD, AB, BC, CD`.
- The shared `distillation_sequencing_env` chemical environment uses only the Python standard library. PyTorch is the sole required learner dependency; do not add Gymnasium or imports from other learner projects.
- Keep deterministic sharp splits, undiscounted negative annual-cost rewards, on-policy PPO updates, and pure greedy evaluation.
- Give every file separate English metadata lines. Give every function and class a multiline English docstring listing inputs and returns.
- Separate logical blocks with blank lines. Add an English intent comment before any logical code block longer than five lines.
- Keep Python files at or below 500 lines and functions at or below 50 lines. Write a unit test for every function.
- After each stage, run focused tests, full pytest with at least 90% package coverage, and mini-linter with warnings treated as failures. After a failed gate, fix the issue and restart from the full test gate.
- Write the stage report in `.agents/reports/` only after all checks pass, then pause for user review before proceeding.
