<!--
Last modified time: 2026-09-23
Last modified content: Record the validated masked REINFORCE policy core
Last modified by: OpenAI Codex
File design: Stage-two implementation and verification report
File purpose: Document policy behavior, gradient checks, quality gates, and the next stage
File creator: OpenAI Codex
-->

# Stage 2: masked REINFORCE core

## Changes

- Added a CPU PyTorch `6 -> 32 -> 10` ReLU policy network. The six inputs preserve the established chemical state order; the ten outputs correspond to textbook tower IDs `1..10`.
- Added a masked categorical distribution. Invalid towers have exactly zero probability and zero policy gradient. Agent methods also reject a mask that does not match its chemical state.
- Added seeded stochastic action sampling with differentiable log probabilities, deterministic legal argmax with smallest-ID tie handling, and read-only action probability reporting.
- Added immutable step and episode records, validation of three connected sharp splits, and undiscounted reward-to-go.
- Added one Adam update using `-(1/B) sum_i sum_t G[i,t] log pi(a[i,t] | s[i,t])`, with default learning rate `1e-3` and intended batch size of 32 complete current-policy episodes. The implementation has no critic, baseline, replay buffer, or exact-solution training labels.
- Exported the policy and agent APIs and updated the README and Agent context. The stage-one chemical environment remains unchanged.

## Focused evidence

The new tests check network output dimensions and gradients; mask normalization, sampling, and blocked-logit gradients; configuration validation; `reward_to_go`; seeded action reproducibility; deterministic ties; trajectory continuity and invalid records; exact loss calculation; batch averaging; and an end-to-end episode sampled from the chemical environment followed by one optimizer step.

## Final verification

Validation ran in the project's own `.venv` with Python 3.12.10 and PyTorch 2.14.0+cpu.

1. Focused tests: `.venv\Scripts\python.exe -m pytest tests\test_policy.py tests\test_reinforce.py -q --no-cov` -> **33 passed**.
2. Full suite and coverage: `.venv\Scripts\python.exe -m pytest -q` -> **89 passed; 99.60% package coverage** (252/253 statements).
3. Style gate: `.venv\Scripts\mini-linter.exe check . --fail-on warning` -> **0 errors, 0 warnings**.

PyTorch emitted one nonfatal import warning because optional NumPy is absent from this environment. The project does not use NumPy; all tests passed. The one uncovered line is a defensive rejection of a nonterminal final state after three otherwise connected legal splits, which cannot occur in this fixed four-component problem.

## Stage boundary

The policy core is ready for online training, but this stage does not yet contain the 20,000-episode training loop, final greedy economic evaluation, or JSON command line entry point. Those remain in stage three. Await user review before continuing.
