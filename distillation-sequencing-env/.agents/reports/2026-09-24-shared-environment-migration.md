<!--
Last modified time: 2026-09-24
Last modified content: Record shared-environment migration and verification
Last modified by: OpenAI Codex
File design: Final migration quality report
File purpose: Summarize shared chemistry, compatibility, results, and timing evidence
File creator: OpenAI Codex
-->

# Shared distillation environment migration

## Changes

- Centralized the fixed Example 17.3 column specifications, immutable domain models, action masks, deterministic sharp splits, and negative annual-cost rewards in `distillation_sequencing_env`.
- Made Tabular Q-learning, DQN, REINFORCE, and PPO depend on version `0.1.0` of the shared standard-library package while preserving their prior public import paths as compatibility exports.
- Installed the shared package into four separate virtual environments; created the PPO virtual environment that was absent before migration. All four `pip check` commands reported no broken requirements.

## Verification

- Focused chemical and compatibility tests: 53 passed in the shared package; 54 passed in each learner project.
- Shared chemistry tests cover all ten columns, feasible actions and masks, illegal actions, all six complete sequences, and the `3.308330 M$/yr` optimum.
- Each project passed full pytest with coverage and mini-linter with 0 errors and 0 warnings:

| Project | Full tests | Coverage | 1,000-episode median before → after |
|---|---:|---:|---:|
| Shared environment | 56 passed | 100.00% | N/A |
| Tabular Q-learning | 103 passed | 98.46% | 0.361 s → 0.358 s |
| DQN | 119 passed | 99.04% | 17.800 s → 16.713 s |
| REINFORCE | 109 passed | 96.84% | 11.217 s → 8.365 s |
| PPO | 133 passed | 97.53% | 12.736 s → 7.991 s |

- Seed-42 CLI output for three 1,000-episode runs and one default run per algorithm matched the pre-migration JSON exactly, including policies, action orders, costs, and rewards. Default action orders stayed `(2,8,10)` for Tabular, DQN, and REINFORCE, and `(2,10,8)` for PPO.
- Three independent 1,000-episode timings were measured before and after. Every median stayed within the accepted slowdown bound of 15% and 0.1 s; all measured medians were lower after migration. Timing can vary with local process and filesystem load.
- PPO had no virtual environment before migration, so its baseline used the existing REINFORCE Python environment with PPO on `PYTHONPATH`; after migration it used its own newly installed environment. Both used PyTorch 2.14.0 and produced identical JSON.
- DQN, REINFORCE, and PPO pytest emitted the previously observed optional PyTorch/NumPy initialization warning. It did not affect any test or training result; mini-linter reported zero warnings.
