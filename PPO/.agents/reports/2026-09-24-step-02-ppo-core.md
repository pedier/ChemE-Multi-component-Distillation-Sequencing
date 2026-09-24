<!--
Last modified time: 2026-09-24
Last modified content: Record verified delivery of the masked PPO learning core
Last modified by: OpenAI Codex
File design: Stage-two implementation and validation report
File purpose: Document actor-critic behavior, tests, coverage, and the remaining stage
File creator: OpenAI Codex
-->

# Stage 2: PPO learning core

## Changes

- Added separate `6 → 32 → 10` actor and `6 → 32 → 1` critic networks using PyTorch only in the learner layer.
- Added a masked categorical distribution with exactly zero probability on infeasible textbook towers and deterministic smallest-ID tie breaking for pure decisions.
- Added immutable action samples, physical rollout steps, complete three-step episodes, and validation of masks, costs, transitions, terminal boundaries, and policy version.
- Added undiscounted GAE with terminal value zero, frozen behavior-policy log probabilities, PPO's signed clipped surrogate, critic mean-squared error, and a training-only entropy term.
- Added batch-level advantage normalization, shuffled multi-epoch minibatches, Adam updates, gradient clipping, optimization diagnostics, and stale-episode rejection after each update.
- Exported the new public interfaces and documented the core separately from the future training loop.

## Verification

Checks used Python 3.12.10 and PyTorch 2.14.0+cpu from the adjacent REINFORCE development environment. This was test tooling only; the PPO package has no runtime imports from that project.

1. Focused PPO tests: `pytest tests/test_policy.py tests/test_ppo_math.py tests/test_ppo_agent.py -q --no-cov` -> **57 passed**.
2. Full suite and coverage: `pytest -q` -> **113 passed; 99.45% package coverage** (363/365 statements).
3. Style gate: `mini-linter check . --fail-on warning` -> **0 errors, 0 warnings**.

PyTorch emitted one nonfatal import warning because optional NumPy is absent in the development environment. The PPO implementation does not import or require NumPy.

## Algorithm acceptance and next stage

Tests verify illegal-action probability zero, hand-calculated GAE targets, clipping for positive and negative advantages, detached old policy statistics, actor-critic gradients, multi-epoch updates, and rejection of stale or physically invalid rollouts. Environment reward remains the negative undiscounted annual cost.

The online 20,000-episode training loop, default-seed optimum check, pure-policy evaluation, learning summary, and JSON command line interface belong to stage three. This report closes stage two for user review before that work begins.
