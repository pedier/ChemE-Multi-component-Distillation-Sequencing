<!--
Last modified time: 2026-09-24
Last modified content: Consolidate problem documentation and reproducible validation
Last modified by: OpenAI Codex
File design: Sequencing documentation
File purpose: Describe verified behavior, operation, and research boundaries
File creator: OpenAI Codex
-->

# Distillation Dueling Double DQN

This independent Python 3.12 project implements a masked Dueling Double DQN for the four-component distillation sequence in *Systematic Methods of Chemical Process Design*, Example 17.3. It uses the shared `distillation_sequencing_env` package and does not import another algorithm project.

## Shared problem

See the [canonical problem specification](../docs/problem.md) for all chemical data, units, legal states, and benchmark trajectories. Chemistry is provided by `distillation_sequencing_env`; this learner imports no other algorithm.

## Learning and evaluation

`DuelingQNetwork` maps six state entries through a `6 → 64 → 64` ReLU trunk to separate scalar-value and ten-action advantage heads, combining them as `Q=V+A−mean(A)`.

`DuelingDoubleDQNAgent` uses CPU networks, seeded legal-action epsilon-greedy selection, a bounded uniform ring replay buffer (oldest-first logical indexing, sampling without a full-buffer copy), a frozen target network, Huber loss, Adam, gradient clipping, and periodic hard synchronization. The online network selects only feasible next towers; the target network evaluates that selection. Terminal transitions do not bootstrap. Training data come solely from online environment interaction, never from an exact-solution label.

Defaults are `10,000` episodes, seed `42`, learning rate `1e-3`, replay capacity `10,000`, batch size `64`, warmup `128` transitions, target sync every `100` optimization steps, and gradient-norm limit `10`. Exploration begins at `1.0`, multiplies by `0.995` after each episode, and has a `0.05` floor. Every episode must terminate after exactly three valid tower selections.

Pure evaluation always chooses the best feasible tower, choosing the lowest ID among values within absolute tolerance `1e-12` of the maximum. It neither explores nor changes weights or replay. With the default configuration, the trained pure policy selects the optimal tower set `{2, 8, 10}` and evaluates to `3.308330 M$/yr` with total reward `-3.308330`. Neural Q estimates are not asserted equal to the exact dynamic-programming table.

## Install and run

From this directory, using Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ..\distillation-sequencing-env
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pip install "..\Tabular Q-Learning\mini-linter.zip"
.\.venv\Scripts\python.exe -m distillation_dqn
```

The CLI prints seeded JSON containing the full configuration, a pure policy for seven reachable nonterminal states, twelve legal state-action Q estimates, and an evaluation record with action order, normalized tower set, selection vector, total cost, and total reward. To run a shorter exploratory experiment or choose another seed:

```powershell
.\.venv\Scripts\python.exe -m distillation_dqn --episodes 500 --seed 7
```

A shorter or different-seed run is not guaranteed to recover the textbook optimum.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\mini-linter.exe check . --fail-on warning
```

The mini-linter is a separately installed development tool. Full verification evidence and the installed PyTorch version are recorded in `.agents/reports/`.


## Compatibility and reproducibility

The old `data`, `environment`, and shared `models` imports are thin compatibility exports. `EvaluationResult` now has one definition in the shared package. State dimensions and episode length come from shared constants but remain fixed at four components.

Repeatability requires the same seed and execution environment; it is not a cross-platform guarantee. Run independent neural experiments in separate processes. See the [root verification guide](../README.md#verification) for the shared chemistry suite and package-specific coverage/linter gates, and the [scaling protocol](../docs/component-scaling.md) for future work.
