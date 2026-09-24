<!--
Last modified time: 2026-09-23
Last modified content: Add REINFORCE-specific chemistry and quality review checks
Last modified by: OpenAI Codex
File design: Stage review checklist
File purpose: Prevent omissions in chemical semantics, policy gradients, tests, and reporting
File creator: OpenAI Codex
-->

# Review checklist

- [ ] Textbook tower IDs, split products, feed flow, heat duty, and cost parameters agree with Example 17.3.
- [ ] Rewards are negative annualized tower costs in M$/yr and use `gamma=1`.
- [ ] Masks exclude every infeasible tower and align at index `action_id - 1`.
- [ ] Three feasible actions complete every episode; invalid actions leave the state unchanged.
- [ ] Once implemented, REINFORCE uses sampled on-policy trajectories and unnormalized reward-to-go without a baseline.
- [ ] Every function has a unit test; total package coverage is at least 90%.
- [ ] File metadata, multiline docstrings, code-block comments, and size limits meet project rules.
- [ ] Focused tests, full pytest with coverage, and mini-linter pass with zero warnings.
- [ ] The current stage report records changes, evidence, and any limitations.
