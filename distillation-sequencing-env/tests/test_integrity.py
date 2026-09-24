"""
Last modified time: 2026-09-24
Last modified content: Verify shared contracts, strict inputs and seeded cleanup behavior
Last modified by: OpenAI Codex
File design: Focused regression tests
File purpose: Preserve chemistry and seeded behavior during refactoring
File creator: OpenAI Codex
"""

from dataclasses import replace
from itertools import product

import pytest

from distillation_sequencing_env import (
    ACTION_COUNT, EPISODE_STEPS, STATE_SIZE, INITIAL_STATE, TERMINAL_STATE,
    DistillationSequenceEnvironment, action_mask, active_mixtures, apply_action,
    available_actions, calculate_column_economics, encode_state,
)
from distillation_sequencing_env.data import COLUMN_BY_ACTION, COMPONENT_MOLE_FRACTIONS
from distillation_sequencing_env.environment import validate_reward
from distillation_sequencing_env.evaluation import evaluate_greedy, policy_records, reachable_states


@pytest.mark.parametrize("state", [None, [1, 0, 0, 0, 0, 0], (True, 0, 0, 0, 0, 0),
                                  (1.0, 0, 0, 0, 0, 0), (1, 0, 0, 1, 0, 0)])
def test_strict_state_boundary(state) -> None:
    """Reject wrong state types and overlapping physical streams.

    Inputs:
        state: Invalid public input, including numerically equal noninteger flags.
    Returns:
        None. Every state reader rejects the malformed state.
    """

    for reader in (active_mixtures, available_actions, action_mask):
        with pytest.raises(ValueError):
            reader(state)


@pytest.mark.parametrize("action", [True, 1.0, "1", None, 0, 11])
def test_strict_action_boundary(action) -> None:
    """Reject noninteger or unknown actions without advancing an episode.

    Inputs:
        action: Invalid external tower identifier.
    Returns:
        None. Readers and stateful transitions preserve their contract.
    """

    environment = DistillationSequenceEnvironment()
    for operation in (calculate_column_economics, environment.step):
        with pytest.raises(ValueError):
            operation(action)
    with pytest.raises(ValueError):
        apply_action(INITIAL_STATE, action)
    assert environment.step(2).info.selected_actions == (2,)


def test_partition_space_and_read_only_constants() -> None:
    """Check all binary encodings and prevent stale cost caches through mutation.

    Inputs:
        Every six-bit vector and the shared textbook mappings.
    Returns:
        None. Exactly eight valid states remain and constants reject writes.
    """

    valid = []
    for state in product((0, 1), repeat=STATE_SIZE):
        try:
            active_mixtures(state)
        except ValueError:
            continue
        valid.append(state)
    assert set(valid) == {*reachable_states(), TERMINAL_STATE}
    assert (STATE_SIZE, ACTION_COUNT, EPISODE_STEPS) == (6, 10, 3)
    with pytest.raises(ValueError, match="overlap"):
        encode_state(("AB", "BC"))
    with pytest.raises(TypeError):
        COMPONENT_MOLE_FRACTIONS["A"] = 0.2
    with pytest.raises(TypeError):
        COLUMN_BY_ACTION[1] = COLUMN_BY_ACTION[2]


@pytest.mark.parametrize("reward", [0.0, float("nan"), float("inf")])
def test_reward_rejects_inconsistent_cost(reward: float) -> None:
    """Reject finite wrong rewards as well as nonfinite observations.

    Inputs:
        reward: Value inconsistent with the cost of tower two.
    Returns:
        None. Reward validation fails before learning consumes the sample.
    """

    with pytest.raises(ValueError):
        validate_reward(2, reward)
    validate_reward(2, -calculate_column_economics(2).annual_cost_musd_per_year)


def test_shared_evaluation_and_policy_inspection() -> None:
    """Verify legal deterministic evaluation and one inspection per graph state.

    Inputs:
        A policy selecting the first feasible action.
    Returns:
        None. Traversal order, complete result, and callback count are correct.
    """

    result = evaluate_greedy(lambda state, mask: available_actions(state)[0],
                             DistillationSequenceEnvironment())
    assert result.action_sequence == (1, 4, 8)
    assert result.total_reward == -result.total_cost_musd_per_year
    assert reachable_states(canonical=True)[0] == INITIAL_STATE
    seen = []

    def inspect(state, mask):
        """Inspect a uniform feasible policy without random sampling.

        Inputs:
            state: Valid nonterminal state.
            mask: Its legal-action mask.
        Returns:
            Uniform probabilities and the smallest legal action.
        """

        seen.append(state)
        return tuple(float(bit) / sum(mask) for bit in mask), available_actions(state)[0]

    records = policy_records(inspect)
    assert tuple(seen) == reachable_states()
    assert len(records) == 7


@pytest.mark.parametrize("early", [True, False])
def test_evaluation_rejects_broken_horizon(early: bool) -> None:
    """Exercise premature and missing termination in the shared evaluator.

    Inputs:
        early: Whether to force early termination or suppress all termination.
    Returns:
        None. Evaluation refuses both invalid episode boundaries.
    """

    environment = DistillationSequenceEnvironment()
    original_step = environment.step

    def broken_step(action):
        """Alter only the terminal signal of an otherwise genuine transition.

        Inputs:
            action: Feasible selected tower.
        Returns:
            Step result with an intentionally inconsistent boundary signal.
        """

        return replace(original_step(action), terminated=early)

    environment.step = broken_step
    with pytest.raises(RuntimeError, match="exactly three"):
        evaluate_greedy(lambda state, mask: available_actions(state)[0], environment)
