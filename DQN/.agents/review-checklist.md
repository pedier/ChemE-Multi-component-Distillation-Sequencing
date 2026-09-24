<!--
Last modified time: 2026-09-21-02:01
Last modified content: Tailor stage review checks to the DQN implementation
Last modified by: OpenAI Codex
File design: Stage review checklist
File purpose: Prevent omissions in chemical, algorithmic, test, and reporting requirements
File creator: OpenAI Codex
-->

# Review Checklist

- [ ] Textbook tower IDs, split tasks, and parameters are unchanged.
- [ ] Reward remains negative undiscounted annualized cost in M$/yr.
- [ ] Invalid actions are masked and rejected by the environment.
- [ ] State transitions are deterministic and every complete flowsheet has three towers.
- [ ] New learner code uses legal next actions and no terminal bootstrap.
- [ ] Every function has a corresponding unit test and total coverage is at least 90%.
- [ ] Python metadata, docstrings, comments, file lengths, and function lengths follow `AGENTS.md`.
- [ ] Focused tests, full pytest/coverage, and mini-linter all pass.
- [ ] The stage result is recorded in `.agents/reports/` before user review.
