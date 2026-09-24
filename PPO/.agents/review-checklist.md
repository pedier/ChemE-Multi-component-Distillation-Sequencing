<!--
Last modified time: 2026-09-24
Last modified content: Define PPO chemistry, learning, and quality review checks
Last modified by: OpenAI Codex
File design: Stage review checklist
File purpose: Prevent omissions in chemical semantics, PPO updates, tests, and reporting
File creator: OpenAI Codex
-->

# Review checklist

- [ ] Textbook tower IDs, split products, feed flow, heat duty, and cost parameters agree with Example 17.3.
- [ ] Rewards are negative annualized tower costs in M$/yr and use `gamma=1`.
- [ ] Masks exclude infeasible towers and align at index `action_id - 1`.
- [ ] Three feasible actions complete every episode; invalid actions leave the state unchanged.
- [ ] Once implemented, PPO uses fresh on-policy trajectories, GAE, and a clipped actor objective.
- [ ] The critic and entropy terms do not change the reported chemical reward.
- [ ] Every function has a unit test; package coverage is at least 90%.
- [ ] Metadata, English multiline docstrings, block comments, and size limits meet project rules.
- [ ] Focused tests, full pytest with coverage, and mini-linter pass with zero warnings.
- [ ] The stage report records changes, evidence, and limitations.
