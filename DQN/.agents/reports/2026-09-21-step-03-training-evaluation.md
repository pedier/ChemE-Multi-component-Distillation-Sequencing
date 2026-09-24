<!--
Last modified time: 2026-09-21-03:05
Last modified content: Record verified complete DQN training, evaluation, and CLI delivery
Last modified by: OpenAI Codex
File design: Final stage delivery and verification report
File purpose: Document training behavior, textbook acceptance, coverage, and known limits
File creator: OpenAI Codex
-->

# Stage 3: Training, pure evaluation, and JSON experiment

## Delivered

- Added default 10,000-episode online training with seed 42. The legal-action exploration probability is fixed within each episode, starts at 1.0, decays by 0.995 per episode, and stops at 0.05.
- Required exactly three valid sharp splits per training and evaluation episode. Premature or missing completion raises a clear error before an invalid transition enters replay.
- Added pure greedy evaluation that does not explore, update weights, consume the exploration random state, or change replay. It reports ordered tower actions, normalized tower set, selection vector, terminal state, cost, and reward.
- Added `python -m distillation_dqn` with `--episodes` and `--seed` overrides. It emits deterministic JSON containing the complete configuration, a policy over seven reachable nonterminal states, twelve legal Q estimates, and the greedy evaluation.
- Updated public exports, user documentation, and agent context. The adjacent Tabular Q-Learning project remains unchanged and is not imported.

## Default textbook acceptance

The default CPU experiment used 10,000 episodes and seed 42. Both the integration test and an independent default CLI run produced:

| Field | Result |
|---|---:|
| Ordered actions | `2 → 8 → 10` |
| Normalized tower set | `{2, 8, 10}` |
| Pure products | Terminal six-zero state |
| Annualized total cost | `3.308330 M$/yr` |
| Total reward | `-3.308330` |
| JSON policy / legal Q records | `7 / 12` |

The final two independent splits may be reversed without changing the flowsheet. Acceptance checks the selected structure and exact environment economics, not equality of every neural Q estimate to a dynamic-programming value. All replay data arise from online environment interactions; no exact-solution labels are used for learning.

## Verification

1. Short focused training, CLI, network, and structure tests after final code cleanup: `29 passed`, with two long-training tests intentionally deselected. Earlier focused training/CLI tests passed `21/21`; the default long-training tests passed `2/2` separately.
2. Final full pytest with coverage: `118 passed`; package line coverage `99.29%`, above the required `90%`.
3. Mini-linter with warnings treated as failures: `0 errors`, `0 warnings`, `0 info`.
4. Default command-line smoke test completed successfully and confirmed the results above. The thin `__main__.py` launcher is covered by this subprocess/smoke verification rather than in-process coverage instrumentation; every other runtime module is at `100%` line coverage.
5. Structural checks confirmed English metadata, comments, and multiline `Inputs:`/`Returns:` docstrings, along with the 500-line file and 50-line function limits.

The run used Python `3.12.10` and PyTorch `2.14.0` in the DQN project's own virtual environment. PyTorch emits one non-failing warning because optional NumPy interoperation is unavailable (`numpy` is not installed). The implementation does not call NumPy, the full test suite passed, and mini-linter reported no warnings.

## Limits

This version solves the fixed, deterministic four-component textbook instance only. It does not train across changing feed compositions, tower costs, energy prices, or component counts. The neural approximation is a method comparison against the exact small-problem benchmark, not an efficiency improvement over enumeration or dynamic programming for this instance.
