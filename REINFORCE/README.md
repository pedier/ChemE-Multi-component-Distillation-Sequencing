<!--
Last modified time: 2026-09-24
Last modified content: Consolidate problem documentation and reproducible validation
Last modified by: OpenAI Codex
File design: Sequencing documentation
File purpose: Describe verified behavior, operation, and research boundaries
File creator: OpenAI Codex
-->

# Four-component distillation sequencing with REINFORCE

This standalone project solves Example 17.3 of *Systematic Methods of Chemical Process Design*. The three stages now provide the chemical environment, PyTorch REINFORCE learner, seeded online training, pure-policy evaluation, and a deterministic JSON command line experiment.

## Shared problem

See the [canonical problem specification](../docs/problem.md) for all chemical data, units, legal states, and benchmark trajectories. Chemistry is provided by `distillation_sequencing_env`; this learner imports no other algorithm.

## REINFORCE policy core

`PolicyNetwork` maps the six state indicators through a `6 -> 32 -> 10` ReLU network. `masked_distribution` assigns exact zero probability to towers without their required feed. `ReinforceAgent.sample_action()` returns a textbook tower ID and its differentiable log probability; `greedy_action()` resolves an exact tie in favor of the smaller tower ID. The agent verifies that a supplied mask matches the chemical state.

`EpisodeTrajectory` records three connected, legal sharp splits. `reward_to_go()` computes each undiscounted suffix sum. `ReinforceAgent.update()` takes one Adam step on the mean of complete-episode losses:

`L = -(1/B) * sum_i sum_t G[i,t] * log pi(a[i,t] | s[i,t])`.

The default learning rate is `1e-3`, the intended update batch is 32 complete episodes, and the seed is 42. The learner has no critic, learned baseline, replay buffer, or exact-solution labels. The training loop collects fresh on-policy batches before each update; the default 20,000 episodes make 625 updates.

## Training and evaluation

Run the default seed-42 experiment or override its episode budget and seed:

```powershell
.\.venv\Scripts\python.exe -m distillation_reinforce
.\.venv\Scripts\python.exe -m distillation_reinforce --episodes 20000 --seed 42
```

`train(config=None, episodes=20_000)` returns the trained agent. Each episode resets a fresh chemical environment, samples exactly three legal towers under unchanged policy weights, and records their undiscounted returns. The agent updates after 32 complete episodes; a final smaller batch is updated when a custom episode count is not divisible by 32.

`evaluate(agent)` chooses the most probable legal tower at each state without sampling, gradient updates, or random-state changes. `policy_snapshot(agent)` reports the ten masked probabilities at every reachable nonterminal state. `training_summary(agent)` reports the first and last 100-episode mean rewards and consecutive 1,000-episode means.

The JSON output contains `algorithm`, `config`, `training`, `policy`, and `evaluation`. The default seed-42 run returns tower sequence `2 -> 8 -> 10`, normalized set `{2, 8, 10}`, annualized cost `3.308330 M$/yr`, and reward `-3.308330`. Tower orders `2 -> 8 -> 10` and `2 -> 10 -> 8` describe the same flowsheet.

## Setup and checks

On Windows with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ..\distillation-sequencing-env
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pip install .\mini-linter.zip
.\.venv\Scripts\python.exe -m pytest tests\test_training.py tests\test_cli.py -q --no-cov
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\mini-linter.exe check . --fail-on warning
```

The chemical-environment API includes `DistillationSequenceEnvironment.reset()`, `step(action_id)`, `available_actions(state)`, `action_mask(state)`, `apply_action(state, action_id)`, and `calculate_column_economics(action_id)` in `distillation_reinforce`. The environment uses the Python standard library; PyTorch is required only by the policy core. The common model comes from the sibling `distillation-sequencing-env` package, without importing another algorithm project. The public learning interface also exports `train`, `evaluate`, `reachable_states`, `policy_snapshot`, and `training_summary`.


## Compatibility and reproducibility

The old `data`, `environment`, and shared `models` imports are thin compatibility exports. `EvaluationResult` now has one definition in the shared package. State dimensions and episode length come from shared constants but remain fixed at four components.

Repeatability requires the same seed and execution environment; it is not a cross-platform guarantee. Run independent neural experiments in separate processes. See the [root verification guide](../README.md#verification) for the shared chemistry suite and package-specific coverage/linter gates, and the [scaling protocol](../docs/component-scaling.md) for future work.

`inspect_policy(state, mask)` returns probabilities and the greedy action from one inference. The existing `action_probabilities` and `greedy_action` methods remain available. Stored trajectory rewards are checked against the selected tower cost before updates.
