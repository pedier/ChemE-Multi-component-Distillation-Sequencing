<!--
Last modified time: 2026-09-23
Last modified content: Record validated training, evaluation, and command line delivery
Last modified by: OpenAI Codex
File design: Stage-three implementation and verification report
File purpose: Document the final REINFORCE workflow and its quality gates
File creator: OpenAI Codex
-->

# Stage 3: training and evaluation

## Changes

- Added online training with a fresh deterministic chemical environment for each episode, legal masked sampling, full three-step trajectories, and one policy update per 32 episodes. A final partial batch is updated as well.
- Added pure greedy evaluation without action sampling or optimizer updates. The result records the tower sequence, tower set, annualized cost, and undiscounted return.
- Added a snapshot of all seven reachable nonterminal states, including each legal action mask, ten action probabilities, and the chosen tower.
- Added learning-curve summaries and a JSON command line entry point with `--episodes` and `--seed`. Defaults are 20,000 episodes and seed 42.
- Updated package exports, usage documentation, and project context. Training continues to use no critic, return baseline, replay buffer, or exact-solution labels.

## Verification

All checks ran in the project's `.venv` with Python 3.12.10 and PyTorch 2.14.0+cpu, in the required order.

1. Focused tests: `.venv\Scripts\python.exe -m pytest tests\test_training.py tests\test_cli.py -q --no-cov` -> **19 passed**.
2. Full suite and coverage: `.venv\Scripts\python.exe -m pytest -q` -> **108 passed; 97.80% package coverage** (356/364 statements).
3. Style gate: `.venv\Scripts\mini-linter.exe check . --fail-on warning` -> **0 errors, 0 warnings**.

The tests cover training configuration, complete-episode batches and partial batches, read-only greedy evaluation, reachable-state policy reporting, learning summaries, argument validation, and command line JSON. PyTorch emitted one nonfatal import warning because optional NumPy is absent; the implementation does not use NumPy.

## Default-seed engineering acceptance

The complete default run (`train()` followed by `evaluate(agent)`) returned action sequence `(2, 8, 10)`, tower set `{2, 8, 10}`, annualized cost `3.308330 M$/yr`, and undiscounted return `-3.308330`. It performed 625 optimizer updates. Mean reward improved from `-4.02951725` over the first 100 episodes to `-3.32246230` over the last 100 episodes.

The fixed four-component Example 17.3 REINFORCE implementation is complete and ready for review.
