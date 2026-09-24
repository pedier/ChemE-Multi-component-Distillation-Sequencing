<!--
Last modified time: 2026-09-24
Last modified content: Consolidate problem documentation and reproducible validation
Last modified by: OpenAI Codex
File design: Sequencing documentation
File purpose: Describe verified behavior, operation, and research boundaries
File creator: OpenAI Codex
-->

# Four-component problem specification

The implemented preset follows Example 17.3, Table 17.1 and Figure 17.6 of *Systematic Methods of Chemical Process Design*. This document describes the implemented textbook model; it is not a rigorous simulator or an independently validated equipment design model.

## Assumptions and units

Components are ordered by volatility as `A, B, C, D`. Total feed is `1000 kmol/h`; initial mole fractions are `0.15, 0.30, 0.35, 0.20`. Every legal tower performs a deterministic sharp split of a contiguous mixture. Pure products disappear from the active-mixture state. There is no recycle, heat integration, equipment sharing, or history-dependent cost.

For tower k with feed mixture S:

- `F_k = 1000 * sum(z_i for i in S)` in kmol/h.
- `Q_k = K_k * F_k`, retaining the textbook's scaled duty convention.
- `c_k = alpha_k + beta_k * F_k + (34.0 + 1.3) * Q_k` in k$/yr.
- Reported annual cost is `c_k / 1000` in M$/yr; reward is its negative.

The repository does not establish a standalone physical unit for `Q_k`, `K_k`, or the utility coefficients beyond this combined scaling. Do not label the duty as kW or GJ/h without checking the original source and its unit conversions. The fixed cost `alpha_k` is in k$/yr; the other coefficients must be interpreted consistently with the formula above.

## State and action contract

`State` is a tuple of six integer zero/one entries ordered `ABCD, ABC, BCD, AB, BC, CD`. Booleans, floats, wrong lengths, and mixtures overlapping in components are rejected. Missing components are already pure. The initial state is `(1,0,0,0,0,0)`; termination is `(0,0,0,0,0,0)`.

External actions are integer tower IDs `1..10`. The ten Boolean mask positions use `action_id - 1`. A tower is legal exactly when its feed mixture is active. A transition removes its feed and adds its non-pure products. Invalid actions raise without changing the episode. Calling `step` after termination requires a reset.

The constants `STATE_SIZE=6`, `ACTION_COUNT=10`, and `EPISODE_STEPS=3` describe this fixed preset, not a generic N-component interface. Every complete episode has three actions. All learners use gamma one; there is no terminal bonus or reward normalization that changes the economic objective.

| State's active mixtures | Legal tower IDs |
| --- | --- |
| ABCD | 1, 2, 3 |
| BCD | 4, 5 |
| ABC | 6, 7 |
| AB and CD | 8, 10 |
| CD | 8 |
| BC | 9 |
| AB | 10 |
| None | None |

There are eight reachable states including termination and twelve legal state-action pairs. All other six-bit encodings are invalid overlapping mixtures.

## Candidate towers

| ID | Split | alpha | beta | K | Annual cost (M$/yr) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | A / BCD | 145 | 0.42 | 0.028 | 1.553400 |
| 2 | AB / CD | 52 | 0.12 | 0.042 | 1.654600 |
| 3 | ABC / D | 76 | 0.25 | 0.054 | 2.232200 |
| 4 | B / CD | 38 | 0.14 | 0.040 | 1.357200 |
| 5 | BC / D | 66 | 0.21 | 0.047 | 1.654735 |
| 6 | A / BC | 125 | 0.78 | 0.024 | 1.426760 |
| 7 | AB / C | 44 | 0.11 | 0.039 | 1.233360 |
| 8 | C / D | 58 | 0.19 | 0.044 | 1.016760 |
| 9 | B / C | 37 | 0.08 | 0.036 | 0.915020 |
| 10 | A / B | 112 | 0.39 | 0.022 | 0.636970 |

## Complete trajectories and acceptance

| Execution order | Total cost (M$/yr) |
| --- | ---: |
| 1, 4, 8 | 3.927360 |
| 1, 5, 9 | 4.123155 |
| 2, 8, 10 | 3.308330 |
| 2, 10, 8 | 3.308330 |
| 3, 6, 9 | 4.573980 |
| 3, 7, 10 | 4.102530 |

Six execution orders describe five flowsheet structures. The minimum is tower set `{2,8,10}` at `3.308330 M$/yr`; total reward is `-3.308330`. The two optimal execution orders are structurally equivalent, while fixed-seed regression checks retain each algorithm's original order.

Costs are calculated in binary floating point; six-decimal displayed equality is not a promise that decimal literals have exact binary representations. Economics tests use appropriate tolerances; refactor comparisons on the same runtime compare complete numerical outputs exactly.
