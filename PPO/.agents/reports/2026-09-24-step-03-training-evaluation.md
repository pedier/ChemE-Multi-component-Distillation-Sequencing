<!--
Last modified time: 2026-09-24
Last modified content: Record verified PPO training, pure evaluation, and JSON delivery
Last modified by: OpenAI Codex
File design: Stage-three implementation and validation report
File purpose: Document final workflow, quality gates, and textbook optimum
File creator: OpenAI Codex
-->

# Stage 3: training and evaluation

## Changes

- Added online training from fresh three-step on-policy episodes, one PPO update per 32 complete episodes, and a final update for any partial batch.
- Added deterministic pure-policy evaluation without action sampling or parameter updates. Its result contains action order, canonical tower set, selection vector, final state, true annualized cost, and total reward.
- Added breadth-first enumeration of all seven reachable nonterminal states, masked probability snapshots, and learning-curve summaries with the last PPO update diagnostics.
- Added a JSON command line interface with `--episodes` and `--seed`, exported the training interfaces, completed the README, and included the local mini-linter development installer.

## Verification

Tests used Python 3.12.10, PyTorch 2.14.0+cpu, pytest 9.1.1, and the mini-linter from the adjacent REINFORCE development environment. The PPO package has no runtime imports from that project. Its included `mini-linter.zip` permits a separate development installation.

1. Focused training and CLI tests: `pytest tests/test_training.py tests/test_cli.py -q --no-cov` -> **19 passed**.
2. Full suite with coverage: `pytest -q` -> **132 passed; 98.11% package coverage** (467/476 statements).
3. Style gate: `mini-linter check . --fail-on warning` -> **0 errors, 0 warnings**.

PyTorch emitted one nonfatal import warning because optional NumPy is absent in the development environment. The PPO implementation does not import or require NumPy.

## Default-seed engineering acceptance

The complete `train()` default run used **20,000 episodes**, seed **42**, and **625 PPO updates**. Pure evaluation returned action order `(2, 10, 8)`, canonical tower set `{2, 8, 10}`, annualized cost `3.308330 M$/yr`, and total undiscounted reward `-3.308330`. The run took 54.25 seconds in the available CPU development environment.

Mean sampled reward improved from `-3.93622115` over the first 100 episodes to `-3.30833000` over the last 100. The final two tower actions commute, so `(2, 10, 8)` is the same optimal flowsheet as `(2, 8, 10)`.

The fixed four-component Example 17.3 PPO implementation is complete and ready for user review.
