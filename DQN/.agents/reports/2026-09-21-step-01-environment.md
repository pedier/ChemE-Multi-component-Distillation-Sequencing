<!--
Last modified time: 2026-09-21-02:03
Last modified content: Record the verified independent DQN chemical environment delivery
Last modified by: OpenAI Codex
File design: Stage-one delivery report
File purpose: Document environment changes, textbook benchmarks, and quality gates
File creator: OpenAI Codex
-->

# Stage 1: Four-component distillation environment

## Delivered

- Created an independent Python 3.12 `distillation_dqn` package in the DQN directory. The adjacent Tabular Q-Learning project was not modified and is not imported.
- Ported the validated textbook data, immutable result models, six-entry state encoding, ten tower actions, action masks, annual-cost calculation, deterministic sharp-split transitions, and auditable episode wrapper.
- Preserved state order `ABCD, ABC, BCD, AB, BC, CD`, textbook action IDs `1..10`, the mask index `action_id - 1`, reward units M$/yr, and exactly three valid splits per flowsheet.
- Added package/test/style configuration, project guidance, English metadata and docstring checks, and the required `.agents` review templates.
- Declared PyTorch for the later DQN learner, but stage-one runtime modules do not import PyTorch. No neural network, replay, or training code has been added yet.

## Chemical benchmarks

| Tower set | Exact annual cost (M$/yr) | Rank |
|---|---:|---:|
| `{2, 8, 10}` | 3.308330 | 1 |
| `{1, 4, 8}` | 3.927360 | 2 |
| `{3, 7, 10}` | 4.102530 | 3 |
| `{1, 5, 9}` | 4.123155 | 4 |
| `{3, 6, 9}` | 4.573980 | 5 |

Tests also cover all ten tower transitions, reachable action sets, invalid actions, flow and heat-duty calculations, termination, reset, and both orders of independent final splits. Every complete path has cumulative reward equal to negative annual cost.

## Verification

1. Focused environment and data tests: `53 passed`.
2. Full pytest with coverage: `56 passed`; package line coverage `100.00%`, above the `90%` minimum.
3. Mini-linter with warnings treated as failures: `0 errors`, `0 warnings`, `0 info`.
4. Structural tests passed: all Python source/test files have English metadata, comments, and docstrings; source callables have `Inputs:` and `Returns:` sections; files and functions meet the line limits.

The tests used Python 3.12.10 and the existing local pytest/mini-linter executables. PyTorch is not installed in that test interpreter and was not needed for this environment-only stage. Stage 2 will create or select a DQN-specific environment and record the installed PyTorch version before testing the neural learner.

## Stage boundary

Stop for user review now. Stage 2 will implement the PyTorch masked Dueling Double DQN core only after approval.
