<!--
Last modified time: 2026-09-23
Last modified content: Record the validated first-stage chemical environment
Last modified by: OpenAI Codex
File design: Stage-one implementation and verification report
File purpose: Document chemistry behavior, quality-gate evidence, and remaining stages
File creator: OpenAI Codex
-->

# Stage 1: four-component chemical environment

## Changes

- Confirmed the target `REINFORCE` directory contained only an empty Git repository before adding files.
- Created a standalone Python 3.12 package named `distillation_reinforce` with textbook candidate-column data, immutable engineering records, cost calculations, deterministic sharp-split transitions, action masks, and episode state.
- Preserved external tower IDs `1..10`, mask index `action_id - 1`, six-state order `ABCD, ABC, BCD, AB, BC, CD`, negative annualized cost rewards in M$/yr, and three-step termination.
- Added project configuration, README, unit tests, style-contract tests, mini-linter archive, and Agent templates. The chemical runtime imports only the Python standard library; the declared PyTorch dependency is for the later policy stage.

## Chemistry checks

The tests calculate flow, heat duty, and annualized cost for all ten textbook towers. They verify all seven reachable nonterminal states, ten tower transition rules, invalid and post-terminal actions, engineering audit records, and all six legal action sequences representing five flowsheets.

| Tower sequence | Annualized cost (M$/yr) |
| --- | ---: |
| 2, 8, 10 | 3.308330 |
| 2, 10, 8 | 3.308330 |
| 1, 4, 8 | 3.927360 |
| 3, 7, 10 | 4.102530 |
| 1, 5, 9 | 4.123155 |
| 3, 6, 9 | 4.573980 |

For every complete sequence, total reward is the negative of the corresponding annualized cost.

## Final verification

All final commands ran in this project's own `.venv` with Python 3.12.10.

1. Focused tests: `.venv\Scripts\python.exe -m pytest tests\test_data.py tests\test_environment.py -q --no-cov` → **53 passed**.
2. Full suite and coverage: `.venv\Scripts\python.exe -m pytest -q` → **56 passed; 100.00% package coverage** (128/128 statements).
3. Style gate: `.venv\Scripts\mini-linter.exe check . --fail-on warning` → **0 errors, 0 warnings**.

The first linter run identified four missing `.agents` templates. They were added, and the full test and linter gates were rerun successfully. The final sequence above passed after all source edits.

## Stage boundary

No policy network, REINFORCE update, training loop, or command line experiment is present yet. The stage-one virtual environment contains the project and test tools; PyTorch is declared in `pyproject.toml` and will be installed and exercised when stage two begins. Await user review before stage two.
