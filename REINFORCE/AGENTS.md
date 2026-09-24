<!--
Last modified time: 2026-09-23
Last modified content: Define REINFORCE-specific chemistry, style, and validation rules
Last modified by: OpenAI Codex
File design: Project-level collaboration contract
File purpose: Preserve textbook semantics and staged quality gates
File creator: OpenAI Codex
-->

# Agent guide

This project implements masked REINFORCE for the fixed four-component distillation sequence problem. Preserve textbook tower IDs `1..10`, action-mask index `action_id - 1`, state order `ABCD, ABC, BCD, AB, BC, CD`, deterministic sharp splits, and undiscounted annual-cost rewards.

- The chemical environment uses only the Python standard library. PyTorch is the sole required runtime dependency for the policy learner; do not introduce Gymnasium.
- Keep REINFORCE on-policy with reward-to-go, `gamma=1`, and no critic, baseline, or exact-solution training labels. Use stochastic masked actions during training and a deterministic legal argmax for evaluation.
- Each Python file has separate English metadata lines. Each function and class has a multiline English docstring listing inputs and returns.
- Separate logical blocks with blank lines. Add an English intent comment before a logical code block longer than five lines.
- Keep Python files at or below 500 lines and functions at or below 50 lines.
- Unit-test every function and maintain package coverage of at least 90%.
- After each stage, run focused tests, full pytest with coverage, then mini-linter with warnings treated as failures. If a gate fails, fix and restart at the full test gate.
- Only after all checks pass, update the stage report in `.agents/reports/` and pause for user review before advancing to the next stage.
