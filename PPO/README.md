<!--
Last modified time: 2026-09-24
Last modified content: Consolidate problem documentation and reproducible validation
Last modified by: OpenAI Codex
File design: Sequencing documentation
File purpose: Describe verified behavior, operation, and research boundaries
File creator: OpenAI Codex
-->

# Four-component distillation sequencing with PPO

This independent Python 3.12 project solves Example 17.3 of *Systematic Methods of Chemical Process Design* using masked clipped PPO. Its chemical model comes from the shared `distillation_sequencing_env` package; its learner remains independent.

## Shared problem

See the [canonical problem specification](../docs/problem.md) for all chemical data, units, legal states, and benchmark trajectories. Chemistry is provided by `distillation_sequencing_env`; this learner imports no other algorithm.

## PPO learner

The actor and critic are separate `6 → 32 → 10` and `6 → 32 → 1` PyTorch networks. The actor samples only legal towers through a masked categorical distribution. Training records each sampled action, old log probability, old value, reward, next state, and terminal flag. Complete three-step trajectories are validated against the chemistry model before each update.

GAE uses `gamma=1` and `lambda=0.95`. The actor uses PPO's signed clipped probability-ratio surrogate. The critic minimizes mean-squared error to GAE value targets. Entropy is a training regularizer only; reported cost always comes from the environment's actual tower costs. Each fresh on-policy batch is reused for four shuffled minibatch epochs, then expired. Pure evaluation selects the highest-probability legal tower with smallest-ID tie breaking.

Defaults are 20,000 episodes, seed 42, Adam learning rate `3e-4`, 32 complete episodes per batch, four update epochs, minibatch size 48, clip radius `0.2`, value coefficient `0.5`, entropy coefficient `0.01`, and gradient-norm limit `0.5`.

## Install and run

From this directory, using Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ..\distillation-sequencing-env
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pip install .\mini-linter.zip
.\.venv\Scripts\python.exe -m distillation_ppo
```

Choose a shorter exploratory run or another seed with `--episodes` and `--seed`:

```powershell
.\.venv\Scripts\python.exe -m distillation_ppo --episodes 500 --seed 7
```

The command prints seeded JSON with `algorithm`, `config`, `training`, `policy`, and `evaluation`. The policy lists all seven reachable nonterminal states and their legal probabilities. Evaluation reports ordered tower actions, the canonical tower set, selection vector, annualized cost, and undiscounted reward. Short or different-seed runs are not guaranteed to recover the optimum.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_training.py tests\test_cli.py -q --no-cov
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\mini-linter.exe check . --fail-on warning
```

Stage reports in `.agents/reports/` record the changes and verification evidence.

## Compatibility and reproducibility

The old `data`, `environment`, and shared `models` imports are thin compatibility exports. `EvaluationResult` now has one definition in the shared package. State dimensions and episode length come from shared constants but remain fixed at four components.

Repeatability requires the same seed and execution environment; it is not a cross-platform guarantee. Run independent neural experiments in separate processes. See the [root verification guide](../README.md#verification) for the shared chemistry suite and package-specific coverage/linter gates, and the [scaling protocol](../docs/component-scaling.md) for future work.

`rollout.py` owns immutable PPO records; existing imports from `ppo.py` remain supported. Reported update diagnostics are arithmetic means over minibatches, including any smaller final minibatch with equal weight. This preserves the existing reporting definition.

`inspect_policy(state, mask)` returns probabilities and the greedy action from one inference. The existing `action_probabilities` and `greedy_action` methods remain available. Stored trajectory rewards are checked against the selected tower cost before updates.
