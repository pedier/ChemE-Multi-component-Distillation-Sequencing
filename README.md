<!--
Last modified time: 2026-09-24
Last modified content: Consolidate problem documentation and reproducible validation
Last modified by: OpenAI Codex
File design: Sequencing documentation
File purpose: Describe verified behavior, operation, and research boundaries
File creator: OpenAI Codex
-->

# Distillation sequencing

Four independent reinforcement-learning implementations solve the same four-component textbook problem. One standard-library package owns the chemistry. The current implementation is fixed at four components; the N-component study is a documented next phase.

## Start here

- [Problem specification](docs/problem.md): data, units, states, actions, rewards, and benchmark.
- [Multi-component research protocol](docs/component-scaling.md): proposed extension and operational failure criteria.
- [Shared environment](distillation-sequencing-env/README.md): API and compatibility boundaries.
- [Tabular Q-learning](Tabular%20Q-Learning/README.md), [DQN](DQN/README.md), [REINFORCE](REINFORCE/README.md), [PPO](PPO/README.md): algorithm-specific usage.
- [Cleanup verification report](.agents/reports/2026-09-24-documentation-cleanup.md): measured before/after results.

| Project | Learner | Default episodes | Runtime |
| --- | --- | ---: | --- |
| Tabular Q-Learning | Masked tabular Q-learning | 10,000 | Python standard library + shared environment |
| DQN | Masked Dueling Double DQN | 10,000 | PyTorch + shared environment |
| REINFORCE | Masked reward-to-go, no baseline | 20,000 | PyTorch + shared environment |
| PPO | Masked clipped PPO with GAE | 20,000 | PyTorch + shared environment |

All default seeds are 42. All use undiscounted negative annual cost. Learners never import one another at runtime. Compatibility modules intentionally preserve established public imports; they are not duplicate chemistry implementations.

## Install

Use Python 3.12 and a separate `.venv` inside each algorithm directory. Run these commands **from that algorithm directory**, replacing the final module name as shown in its README:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ..\distillation-sequencing-env
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pip install "..\Tabular Q-Learning\mini-linter.zip"
.\.venv\Scripts\python.exe -m distillation_q_learning
```

The shared distribution is `distillation-sequencing-env==0.1.0`; its Python import is `distillation_sequencing_env`. Editable installation makes shared changes visible to all four environments. There is no need to merge virtual environments.

## Verification

Run from **each algorithm directory**:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_shared_environment.py tests\test_training.py tests\test_cli.py -q --no-cov
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\mini-linter.exe check . --fail-on warning
.\.venv\Scripts\python.exe -m pip check
```

Run the authoritative chemistry suite separately from **`Sequencing\distillation-sequencing-env`**, using the Tabular environment as a development interpreter:

```powershell
& "..\Tabular Q-Learning\.venv\Scripts\python.exe" -m pytest -q
& "..\Tabular Q-Learning\.venv\Scripts\mini-linter.exe" check . --fail-on warning
```

Each package requires at least 90% coverage and zero mini-linter warnings. Chemistry tests live once in the shared package; learner tests retain compatibility and learning integration checks. Historical reports are point-in-time evidence and may describe the former layout; current READMEs and the latest report describe the current code.

## Reproducibility and comparison

For a behavior-preserving refactor, compare complete JSON outputs, learned parameters, and available training histories, using identical seeds, Python/PyTorch versions, hardware, and thread settings. The four algorithms do not need to learn identical policies at unvisited states. Within each algorithm, the old and new results must agree.

For timing, use the same interpreter for both source versions, separate fresh processes, a 100-episode warmup, and three alternating before/after runs of 1,000 episodes. Time `train()` including agent construction, but exclude imports, serialization, tests, and other competing training jobs. Investigate a median slowdown only when it exceeds both 15% and 0.1 seconds. Do not interpret unrelated cold-start measurements as a speedup.

Seeded repeatability is scoped to the tested execution environment, not guaranteed across platforms or PyTorch releases. REINFORCE and PPO currently use the process-global PyTorch RNG; run independent experiments in independent processes. Long runs retain return histories, and policy reporting enumerates the full four-component graph. Neither behavior should be carried unchanged into large-N experiments.
