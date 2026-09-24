<!--
Last modified time: 2026-09-24
Last modified content: Consolidate problem documentation and reproducible validation
Last modified by: OpenAI Codex
File design: Sequencing documentation
File purpose: Describe verified behavior, operation, and research boundaries
File creator: OpenAI Codex
-->

# Distillation Tabular Q-learning

This Python-standard-library learner solves the fixed four-component problem using masked tabular Q-learning. The sibling `distillation_sequencing_env` package owns the chemistry. See the [problem specification](../docs/problem.md) for states, actions, costs, units, and all benchmark trajectories.

## Learning rule and defaults

The sparse Q table stores only visited legal `(state, action)` pairs; unvisited legal values are zero. Exploration samples uniformly from feasible towers. The update is `Q <- Q + alpha * (reward + gamma * max_legal Q(next_state) - Q)` with no bootstrap at termination. Gamma is fixed to one. Greedy evaluation selects the lowest action ID within absolute tolerance `1e-12` of the maximum Q value.

Defaults: 10,000 episodes, seed 42, learning rate 0.2, epsilon start 1.0, minimum 0.05, and multiplicative decay 0.995 between episodes. Each episode contains exactly three physical transitions. Evaluation neither learns nor consumes exploration randomness. The default trajectory is `2 -> 8 -> 10`, costing `3.308330 M$/yr` with reward `-3.308330`.

The low-level `update` method remains a Bellman arithmetic primitive accepting a caller-supplied reward and successor subject to terminal/action checks; the training loop supplies genuine environment observations. It is not a replacement for environment transition validation.

## Install and run

Run from this directory with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ..\distillation-sequencing-env
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pip install .\mini-linter.zip
.\.venv\Scripts\python.exe -m distillation_q_learning
.\.venv\Scripts\python.exe -m distillation_q_learning --episodes 1000 --seed 7
```

The CLI reports `config`, `policy` (seven nonterminal states), `q_values` (twelve legal pairs), and `evaluation` (ordered actions, canonical tower set, selection vector, cost and reward). Policy and Q rows use depth-then-state ordering. Short runs or other seeds need not recover the optimum.

## API and verification

`train(config=None, episodes=10_000)` returns a `TabularQLearningAgent`; `evaluate(agent)` returns the shared immutable `EvaluationResult`. Existing chemistry imports from `distillation_q_learning`, `data`, `models`, and `environment` remain supported as forwarding exports. Dimensions come from shared constants and remain fixed at four components.

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_q_learning.py tests\test_shared_environment.py -q --no-cov
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\mini-linter.exe check . --fail-on warning
```

Run the shared chemistry suite separately as described in the [root guide](../README.md#verification). Coverage must be at least 90%; linter warnings fail the gate. Reports are in `.agents/reports/`. Reproducibility comparisons require the same seed and software/runtime environment. The [component scaling protocol](../docs/component-scaling.md) describes future experiments; it is not implemented by this fixed preset.
