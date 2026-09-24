<!--
Last modified time: 2026-09-24
Last modified content: Describe the common chemical model and four consumers
Last modified by: OpenAI Codex
File design: Shared package context
File purpose: Explain ownership of textbook data and environment behavior
File creator: OpenAI Codex
-->

# Project context

`src/distillation_sequencing_env/` owns the textbook column data, immutable chemical models, legal action masks, annual cost calculation, and deterministic sharp-split transitions. Runtime uses only the Python standard library.

The sibling Tabular Q-learning, DQN, REINFORCE, and PPO projects each install this package in their own virtual environment. Their former `data`, `environment`, and common `models` modules re-export shared objects for compatibility; learning and evaluation code stays in each algorithm project.

All complete legal trajectories use three towers. The optimal tower set is `{2,8,10}` with annualized cost `3.308330 M$/yr`; the last two independent towers can occur in either order.
