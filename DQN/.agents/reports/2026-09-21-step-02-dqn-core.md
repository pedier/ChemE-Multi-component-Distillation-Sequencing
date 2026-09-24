<!--
Last modified time: 2026-09-21-02:44
Last modified content: Record tested masked Dueling Double DQN core delivery
Last modified by: OpenAI Codex
File design: Stage-two delivery and verification report
File purpose: Document learner behavior, coverage, dependencies, and stage limits
File creator: OpenAI Codex
-->

# Stage 2: Masked Dueling Double DQN core

## Delivered

- Created an isolated Python 3.12 virtual environment in the DQN project and installed PyTorch `2.14.0`, pytest, and pytest-cov. The existing Tabular project and its environment were not modified.
- Added a CPU `6 → 64 → 64` ReLU network with separate scalar-value and ten-action advantage heads; action values use `Q=V+A−mean(A)`.
- Added immutable replay transitions that validate the chosen sharp split, next state, next action mask, termination flag, and finite reward.
- Added bounded, seeded uniform experience replay and independent seeded legal-action epsilon-greedy exploration.
- Added a frozen target network and undiscounted masked Double DQN targets: online values select only legal next actions; target values evaluate that selection; terminal rows have zero bootstrap.
- Added Huber-loss Adam optimization, replay warmup, gradient-norm clipping, and hard target synchronization every configured number of updates.
- Exported the new public APIs and updated the project README and agent context. Complete-episode training, evaluation, and CLI remain outside this stage.

## Verified behavior

- The network has the agreed dimensions and mean-centered dueling identity, with gradient flow through both heads.
- Illegal towers cannot be selected by exploration, greedy inference, or Bellman bootstrap. Equal legal Q values resolve to the smallest textbook tower ID.
- Controlled target tests distinguish online action selection from target-network evaluation and verify that completed flowsheets do not bootstrap.
- Replay rejects impossible transitions, evicts the oldest entry at capacity, and samples reproducibly. Identical seeds produce identical network initialization and action samples without advancing PyTorch's global random state.
- Optimizer tests verify no update before warmup, online weight change after warmup, frozen target weights before the synchronization interval, and exact hard synchronization afterward.

## Quality gates

1. Focused tests for network, replay, targets, and agent: `39 passed`. After strengthening the target-sync assertion, the agent-focused subset passed again (`18 passed`).
2. Final full pytest and coverage: `95 passed`; package line coverage `100.00%`, above the required `90%`.
3. Mini-linter with warnings treated as failures: `0 errors`, `0 warnings`, `0 info`.
4. Structural checks passed: Python file/function size limits, English metadata/comments/docstrings, and `Inputs:`/`Returns:` callable documentation.

PyTorch emitted one non-failing test warning because NumPy is not installed in this deliberately PyTorch-only runtime environment. The DQN code does not call NumPy, and all tests passed. The mini-linter reported no warnings.

## Stage boundary

Stop for user review. Stage 3 will add the 10,000-episode default training loop, epsilon schedule, deterministic pure-policy evaluation, optimal-flowsheet acceptance, and JSON CLI only after approval.
