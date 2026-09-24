<!--
Last modified time: 2026-09-24
Last modified content: Consolidate problem documentation and reproducible validation
Last modified by: OpenAI Codex
File design: Sequencing documentation
File purpose: Describe verified behavior, operation, and research boundaries
File creator: OpenAI Codex
-->

# Shared distillation sequencing environment

Distribution: `distillation-sequencing-env==0.1.0`. Import: `distillation_sequencing_env`. Runtime: Python standard library only. This package is the single source of chemical data, state/action rules, deterministic transitions, economics, and evaluation records for all four learners.

See the [canonical problem specification](../docs/problem.md) for the textbook assumptions, units, all towers and all complete trajectories. See the [root guide](../README.md) for installing this package in each separate learner environment.

## Public interface

- `DistillationSequenceEnvironment.reset()` returns `(state, mask)` and clears history/cost.
- `step(action_id)` returns an immutable `StepResult` with state, reward, termination, next mask and `StepInfo` engineering records. Invalid actions leave the episode unchanged.
- `encode_state`, `active_mixtures`, `available_actions`, `action_mask` and `apply_action` implement the same pure chemical rules without maintaining an episode.
- `calculate_column_economics(action_id)` returns an immutable precomputed `ColumnEconomics` record.
- `State`, `ActionMask`, `ColumnSpec`, `ColumnEconomics`, `StepInfo`, `StepResult`, and `EvaluationResult` are shared contracts.
- `STATE_SIZE`, `ACTION_COUNT`, `EPISODE_STEPS` describe the fixed preset as `6`, `10`, `3`.
- `evaluation.reachable_states(canonical=False)` returns breadth-first discovery order; `canonical=True` sorts by depth then tuple, retaining the historical Q-learning/DQN output order.
- `evaluation.evaluate_greedy` evaluates a pure callback; `evaluation.policy_records` builds probability reports with one inspection per state.

The former chemical import paths in each learner remain forwarding layers. Existing valid four-component calls retain their values and return shapes. Input validation now rejects bool/float action IDs, bool/float state flags and overlapping active streams. Textbook mappings are read-only; mutating global feed or tower data was never a supported way to construct a different instance and now raises `TypeError`. A future configurable model must introduce explicit instances.

The fixed model precomputes eight state/mask records and ten economics records; its successor cache is bounded to twelve legal pairs. No training data or learner state is shared between algorithms. `environment.validate_reward` checks finite observed rewards against exact tower costs with absolute tolerance `1e-9`.

## Verify

From **this directory** after installing the Tabular learner's test and linter dependencies:

```powershell
& "..\Tabular Q-Learning\.venv\Scripts\python.exe" -m pytest tests\test_environment.py tests\test_integrity.py -q --no-cov
& "..\Tabular Q-Learning\.venv\Scripts\python.exe" -m pytest -q
& "..\Tabular Q-Learning\.venv\Scripts\mini-linter.exe" check . --fail-on warning
```

This is the authoritative chemistry test suite. Learner suites additionally test imports, interaction with the environment, and their own algorithms. Each suite has a 90% coverage gate; mini-linter warnings are failures. The shared package does not require its own virtual environment to run these development checks.
