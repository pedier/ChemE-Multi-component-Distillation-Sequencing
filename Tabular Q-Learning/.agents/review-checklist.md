<!--
Last modified time: 2026-09-14-15:04
Last modified content: Translate all metadata comments into English
Last modified by: OpenAI Codex
File design: Stage review checklist
File purpose: Prevent omissions in state, cost, testing, and documentation requirements
File creator: OpenAI Codex
-->

# Review checklist

- [ ] Preserve textbook tower IDs, sharp splits and economic parameters.
- [ ] Use undiscounted negative annual cost as the reward.
- [ ] Reject infeasible actions and inconsistent physical states.
- [ ] Preserve deterministic Markov transitions for the fixed instance.
- [ ] Test functions and maintain at least 90% package coverage.
- [ ] Keep English metadata, docstrings and intent comments within size limits.
- [ ] Pass focused tests, full pytest/coverage, and mini-linter with zero warnings.
- [ ] Compare seeded outputs and runtime with the pre-change source snapshot.
- [ ] Record final evidence in `.agents/reports/` after every gate passes.
