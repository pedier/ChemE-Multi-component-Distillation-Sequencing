<!--
Last modified time: 2026-09-24
Last modified content: Consolidate problem documentation and reproducible validation
Last modified by: OpenAI Codex
File design: Sequencing documentation
File purpose: Describe verified behavior, operation, and research boundaries
File creator: OpenAI Codex
-->

# Component-count research protocol

Status: proposed next phase. The current code still implements only the four-component preset. This phase documents the design; it does not implement a generic environment or establish a measured failure threshold.

## Research question

How does each algorithm's probability of learning a near-optimal feasible flowsheet change with component count, instance family, interaction budget, and compute resources? Train each algorithm separately on each instance. Report a budget-qualified transition region rather than an intrinsic component count at which an algorithm always fails.

Start with reproducible synthetic instances. Vary feed balance, split difficulty and the gap between optimal and competing flowsheets; component count alone is not a complete measure of difficulty. Keep all costs positive, use comparable cost scales, freeze generator versions/seeds, and give every algorithm identical instances. Synthetic economics must be labeled as such, not presented as physically validated predictions. Introduce real chemical property and equipment models only in a later validated phase.

## Environment design

Represent components with integer IDs, mixed streams as intervals `[i,j]`, and towers as cuts `(i,k,j)`. An immutable `ProblemSpec` owns composition, candidate cuts, costs, deterministic action mapping, and an instance fingerprint. Preserve the textbook tower IDs as the four-component preset; larger cases use a documented lexicographic interval/cut ordering.

A complete episode has `N-1` actions. There are `N(N-1)/2` possible mixed intervals and `N(N-1)(N+1)/6` candidate interval/cut actions. Reachable partition states number `2^(N-1)`, including termination. Flowsheet structures number Catalan(N-1); chronological split orders number `(N-1)!`. Different chronological orders can yield the same structure.

Derive input/output dimensions, masks, and horizon from the instance. Start with fixed-size networks trained separately for each N/instance to retain a direct comparison with current algorithms. Record network size and trainable parameter count; a later graph-based or shared action-scoring policy is a separate architecture experiment. Costs/composition need not be state inputs when an agent is trained for one fixed instance, but a policy shared across instances would require instance information.

Use bounded or optional training-history retention. Make full graph enumeration an explicit small-instance diagnostic; default reports should use visited or sampled states. Avoid allocating all partition states and avoid unbounded transition caches.

## Exact reference solution

With contiguous sharp splits and independent additive tower costs, use interval dynamic programming:

`V(i,i) = 0`

`V(i,j) = min_k [c(i,k,j) + V(i,k) + V(k+1,j)]`, for `i <= k < j`.

The reference requires O(N^3) time and O(N^2) value-table memory when a single tower cost is available in constant time. A precomputed action-cost table itself requires O(N^3) storage; report that separately, or generate costs on demand. Composition prefix sums can make interval feed lookup constant time. Retain minimizing cuts to reconstruct an optimal tree.

Increasing N under these assumptions does not make the exact optimization inherently exponential. The experiment measures RL sample and compute efficiency against an available exact optimizer. Evaluate a greedy immediate-cost baseline and a uniform legal random baseline as well. Keep the DP oracle strictly outside RL training, replay, labels, and reward shaping.

Heat integration, coupled constraints, recycle or path-dependent costs can invalidate this recurrence; any such model extension requires a new state definition and a new reference-solution method.

## Experimental defaults

1. Screen N = `4, 6, 8, 10, 12, 16, 20, 24, 32`; add neighboring integer sizes around observed degradation.
2. Use interaction budgets `10^4`, `10^5`, `10^6` transitions. Execute only complete episodes; record actual transitions and the unused remainder. Equal episode counts are not equal interaction budgets when N changes.
3. Screen with 3 instances per family and 3 training seeds. Confirm the boundary with 10 independent instances per family and 10 seeds per instance.
4. Evaluate the final deterministic greedy policy. Success requires a legal complete separation and relative gap `(C-C*)/C* <= 1%`; target at least 90% success. Do not substitute the best trajectory sampled during training.
5. Report success fraction, a 95% interval obtained by resampling instances and then seeds, gap distributions, wall time, transition count, and peak memory. Report aggregate uncertainty without treating dependent seeds on the same instance as independent instances. Mark regions whose interval crosses 90% as uncertain.
6. Separate quality failure, invalid flow, nonfinite numerical failure, timeout and out-of-memory. A manually interrupted or otherwise unfinished run is censored, not silently counted as a completed experiment.
7. Freeze hardware, software, CPU threads and resource caps before each sweep. Pilot cap: one hour wall time per run and 8 GiB process working-set memory, with sequential runs on a machine having at least 16 GiB RAM. Monitor child processes, record why a run stops, and publish any revised cap as a new protocol version.
8. Run a fixed-hyperparameter baseline first. Report equal-budget tuning separately: at most 20 configurations per algorithm/N, using held-out tuning instances and at most 10^5 interactions per configuration/seed. Select on tuning results, freeze settings, then evaluate on untouched test instances. Report tuning cost separately.
9. Use one fresh subprocess per run to isolate Python/PyTorch RNG and global settings. Record initialization, training, evaluation, oracle preprocessing and end-to-end time separately. Store instance fingerprint, seed, configuration, dependency versions and failure category with each result.

## Delivery sequence

1. Implement `ProblemSpec` and generic transitions; prove exact four-component preset equivalence and graph-count identities for small N.
2. Implement and test the DP oracle against exhaustive enumeration for small N. Test ties, skewed compositions and near-degenerate costs.
3. Adapt learner dimensions and episode handling without changing each algorithm's update definition. Keep fixed-seed four-component regression gates.
4. Add isolated experiment execution, resource monitoring, checkpoints, bounded diagnostics and machine-readable records.
5. Run the pilot, freeze the instance generator and protocol, then execute confirmation sweeps and publish curves with uncertainty.

Multiple seeds, multiple tasks and interval estimates address the reliability concerns discussed in [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264) and [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560). No algorithm ranking or failure count is claimed before these experiments are run.
