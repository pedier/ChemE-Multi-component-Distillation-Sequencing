<!--
Last modified time: 2026-09-24
Last modified content: Record completed documentation cleanup and regression validation
Last modified by: OpenAI Codex
File design: Completed verification report
File purpose: Preserve measured correctness, performance and quality evidence
File creator: OpenAI Codex
-->

# Documentation and code cleanup: completed

## Changes

- Added the root guide, canonical problem specification and a concrete N-component research protocol; updated all five package READMEs and current learner contexts in English.
- Centralized immutable evaluation records and deterministic graph/evaluation helpers while retaining public import paths and existing output ordering.
- Precomputed the fixed instance's economics and state/mask records, bounded the successor cache, and made textbook lookup mappings read-only.
- Replaced DQN's full-deque sampling copy with a ring buffer. Regression tests compare sampled object identities and RNG states with the original algorithm, including eviction, wrapping and both random.sample regimes.
- Reused shared dimension constants, avoided repeated Q-value validation, combined probability/greedy reporting into one inference, reused PPO sampling tensors, and separated PPO rollout records from optimization code.
- Strengthened state/action and recorded-reward validation. Existing valid experiments retain their values. New rejections include bool/float action IDs, bool/float state flags, overlapping streams and inconsistent stored tower rewards. The low-level tabular Bellman update retains its documented arithmetic interface.
- Centralized duplicate chemistry tests; retained learner compatibility and integration tests. Fewer tests reflect deduplication, not missing chemical cases.

## Correctness and compatibility

All ten tower economics, all eight valid states, all twelve legal state-action pairs and six complete trajectories pass. The optimum remains tower set {2,8,10} at 3.308330 M$/yr. Both optimal execution orders describe the same flowsheet.

Each learner was compared at its default episode count and at 1,000 episodes, seed 42. Complete parsed CLI JSON matches exactly. Neural state dictionaries and available REINFORCE/PPO return, loss and update histories also match exactly. Tabular Q values are included in the CLI comparison. Each project's original pre-environment-migration default JSON was additionally checked against this cleanup's baseline and matched exactly.

| Learner | Default action order | Cost (M$/yr) | Before/after |
| --- | --- | ---: | --- |
| Tabular Q-Learning | 2 -> 8 -> 10 | 3.308330 | exact |
| DQN | 2 -> 8 -> 10 | 3.308330 | exact |
| REINFORCE | 2 -> 8 -> 10 | 3.308330 | exact |
| PPO | 2 -> 10 -> 8 | 3.308330 | exact |

These are fixed-seed regression results on the same installed runtime, not a claim about all seeds, platforms, or future dependencies. REINFORCE/PPO global RNG behavior was preserved; independent research runs should use separate processes.

## Quality gates

Focused tests passed, followed by complete pytest/coverage and mini-linter with warnings treated as failures. Each project's dependency check also passed. All Python files remain at most 500 lines, all functions at most 50 lines; English metadata/docstring checks pass. All documented tower and trajectory costs were checked programmatically.

| Package | Passed tests | Coverage | Linter errors / warnings |
| --- | ---: | ---: | ---: |
| distillation-sequencing-env | 74 | 100.00% | 0 / 0 |
| Tabular Q-Learning | 53 | 98.15% | 0 / 0 |
| DQN | 70 | 98.94% | 0 / 0 |
| REINFORCE | 57 | 97.27% | 0 / 0 |
| PPO | 81 | 97.65% | 0 / 0 |

Total: 335 tests. Neural pytest processes still emit the pre-existing optional NumPy initialization warning from PyTorch; this is not a mini-linter warning or a failed test. NumPy was not added as a new runtime dependency. Mini-linter 0.1.1 was installed from the existing local archive into DQN's development environment.

## Performance

Each cell is the median of three fresh-process, 1,000-episode training measurements after a 100-episode warmup. Before/after order alternates between repetitions. The same project interpreter and installed dependencies run the original source snapshot and updated source. Timing includes train/agent construction, excludes imports and serialization, and runs without another test or training job launched by this task.

| Learner | Before (seconds) | After (seconds) | Change |
| --- | ---: | ---: | ---: |
| Tabular Q-Learning | 0.065964 | 0.022736 | -65.53% |
| DQN | 9.149115 | 9.104140 | -0.49% |
| REINFORCE | 2.584392 | 2.482857 | -3.93% |
| PPO | 1.583720 | 1.496768 | -5.49% |

All pass the agreed gate: no slowdown exceeds both 15% and 0.1 seconds. Small changes should be treated as timing noise; these measurements are not a broad hardware performance claim. Raw runs record Python, PyTorch and thread settings. Earlier cold-start/migration timing measurements were not used for this gate.

## Evidence and next phase

- [Machine-readable summary](cleanup-evidence/summary.json)
- [Full outputs, logs, timings and execution scripts](cleanup-evidence/raw-validation.zip)
- [Original source and package configurations](cleanup-evidence/before-source.zip)
- [Before/after source hashes](cleanup-evidence/source-sha256.json)
- [N-component protocol](../../docs/component-scaling.md)

Working-tree status was saved before edits. Earlier uncommitted environment migration changes were preserved; no Git commits or repository restructuring were performed. The N-component platform remains unimplemented by design. Its protocol uses independently trained synthetic instances, an exact interval-DP oracle, budget-qualified near-optimal success rates and uncertainty estimates.
