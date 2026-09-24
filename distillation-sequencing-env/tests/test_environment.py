"""
Last modified time: 2026-09-24
Last modified content: Translate metadata, docstrings, and code comments into English
Last modified by: OpenAI Codex
File design: Distillation environment behavior tests
File purpose: Verify every public environment function and textbook cost benchmark
File creator: OpenAI Codex
"""

from __future__ import annotations

import pytest

from distillation_sequencing_env import (
    INITIAL_STATE,
    TERMINAL_STATE,
    DistillationSequenceEnvironment,
    action_mask,
    active_mixtures,
    apply_action,
    available_actions,
    calculate_column_economics,
    encode_state,
)


@pytest.mark.parametrize(
    ("active", "state"),
    [
        (("ABCD",), (1, 0, 0, 0, 0, 0)),
        (("AB", "CD"), (0, 0, 0, 1, 0, 1)),
        (("BC",), (0, 0, 0, 0, 1, 0)),
        ((), TERMINAL_STATE),
    ],
)
def test_state_encoding_round_trip(active: tuple[str, ...], state: tuple[int, ...]) -> None:
    """Verify lossless conversion between active streams and state tuples.

    Inputs:
        active: Names of active mixed streams.
        state: Corresponding six-element binary state.
    Returns:
        None. The test passes when encoding and decoding both match.
    """

    assert encode_state(active) == state
    assert active_mixtures(state) == active


@pytest.mark.parametrize("active", [("XYZ",), ("AB", "AB")])
def test_encode_state_rejects_invalid_mixtures(active: tuple[str, ...]) -> None:
    """Verify that state encoding rejects unknown or duplicate streams.

    Inputs:
        active: Invalid collection of active mixed streams.
    Returns:
        None. The test passes when encoding raises ValueError.
    """

    with pytest.raises(ValueError):
        encode_state(active)


@pytest.mark.parametrize("state", [(1, 0), (1, 0, 0, 0, 0, 2)])
def test_active_mixtures_rejects_invalid_states(state: tuple[int, ...]) -> None:
    """Verify that state decoding rejects bad lengths and non-binary values.

    Inputs:
        state: Invalid state tuple.
    Returns:
        None. The test passes when decoding raises ValueError.
    """

    with pytest.raises(ValueError):
        active_mixtures(state)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (INITIAL_STATE, (1, 2, 3)),
        ((0, 0, 1, 0, 0, 0), (4, 5)),
        ((0, 1, 0, 0, 0, 0), (6, 7)),
        ((0, 0, 0, 1, 0, 1), (8, 10)),
        ((0, 0, 0, 1, 0, 0), (10,)),
        ((0, 0, 0, 0, 1, 0), (9,)),
        ((0, 0, 0, 0, 0, 1), (8,)),
        (TERMINAL_STATE, ()),
    ],
)
def test_available_actions_cover_reachable_states(
    state: tuple[int, ...], expected: tuple[int, ...]
) -> None:
    """Verify that every reachable state enables only columns with present feeds.

    Inputs:
        state: Reachable six-element state.
        expected: Expected feasible action IDs.
    Returns:
        None. The test passes when feasible actions match exactly.
    """

    assert available_actions(state) == expected


def test_action_mask_uses_action_id_minus_one_indexing() -> None:
    """Verify that the length-10 mask aligns with textbook action IDs.

    Inputs:
        State in which AB and CD are both active.
    Returns:
        None. The test passes when only columns 8 and 10 are enabled.
    """

    expected = (False, False, False, False, False, False, False, True, False, True)
    assert action_mask(encode_state(("AB", "CD"))) == expected
    assert action_mask(TERMINAL_STATE) == (False,) * 10


@pytest.mark.parametrize(
    ("action_id", "feed_flow", "heat_duty", "annual_cost"),
    [
        (1, 1000.0, 28.0, 1.553400),
        (2, 1000.0, 42.0, 1.654600),
        (3, 1000.0, 54.0, 2.232200),
        (4, 850.0, 34.0, 1.357200),
        (5, 850.0, 39.95, 1.654735),
        (6, 800.0, 19.2, 1.426760),
        (7, 800.0, 31.2, 1.233360),
        (8, 550.0, 24.2, 1.016760),
        (9, 650.0, 23.4, 0.915020),
        (10, 450.0, 9.9, 0.636970),
    ],
)
def test_column_economics_match_table_17_1(
    action_id: int, feed_flow: float, heat_duty: float, annual_cost: float
) -> None:
    """Verify flow, heat duty, and annual cost for every column.

    Inputs:
        action_id: Textbook column ID.
        feed_flow: Expected feed flow rate.
        heat_duty: Expected heat duty in textbook scaling.
        annual_cost: Expected annual cost in millions of dollars.
    Returns:
        None. The test passes when every value matches within tolerance.
    """

    economics = calculate_column_economics(action_id)
    assert economics.feed_flow_kmol_per_hour == pytest.approx(feed_flow)
    assert economics.heat_duty_q == pytest.approx(heat_duty)
    assert economics.annual_cost_musd_per_year == pytest.approx(annual_cost)
    assert economics.annual_cost_kusd_per_year == pytest.approx(annual_cost * 1000.0)


def test_column_economics_rejects_unknown_action() -> None:
    """Verify that the economics calculation rejects an unknown column ID.

    Inputs:
        Action ID 11.
    Returns:
        None. The test passes when the function raises ValueError.
    """

    with pytest.raises(ValueError, match="未知动作编号"):
        calculate_column_economics(11)


@pytest.mark.parametrize(
    ("state", "action_id", "expected"),
    [
        (INITIAL_STATE, 1, (0, 0, 1, 0, 0, 0)),
        (INITIAL_STATE, 2, (0, 0, 0, 1, 0, 1)),
        (INITIAL_STATE, 3, (0, 1, 0, 0, 0, 0)),
        ((0, 0, 1, 0, 0, 0), 4, (0, 0, 0, 0, 0, 1)),
        ((0, 0, 1, 0, 0, 0), 5, (0, 0, 0, 0, 1, 0)),
        ((0, 1, 0, 0, 0, 0), 6, (0, 0, 0, 0, 1, 0)),
        ((0, 1, 0, 0, 0, 0), 7, (0, 0, 0, 1, 0, 0)),
        ((0, 0, 0, 0, 0, 1), 8, TERMINAL_STATE),
        ((0, 0, 0, 0, 1, 0), 9, TERMINAL_STATE),
        ((0, 0, 0, 1, 0, 0), 10, TERMINAL_STATE),
    ],
)
def test_apply_action_covers_all_tower_transitions(
    state: tuple[int, ...], action_id: int, expected: tuple[int, ...]
) -> None:
    """Verify deterministic sharp-split transitions for all 10 columns.

    Inputs:
        state: State before executing the column.
        action_id: Column ID to execute.
        expected: Expected next state.
    Returns:
        None. The test passes when the transition matches.
    """

    assert apply_action(state, action_id) == expected


@pytest.mark.parametrize("action_id", [0, 11, 8])
def test_apply_action_rejects_unknown_or_infeasible_actions(action_id: int) -> None:
    """Verify that transitions reject unknown and currently infeasible actions.

    Inputs:
        action_id: Unknown column or a column infeasible in the initial state.
    Returns:
        None. The test passes when the transition raises ValueError.
    """

    with pytest.raises(ValueError):
        apply_action(INITIAL_STATE, action_id)


def test_environment_reset_and_step_return_auditable_information() -> None:
    """Verify that reset and step return complete, auditable engineering data.

    Inputs:
        A new distillation environment and action 2.
    Returns:
        None. The test passes when state, reward, cost, and selection data match.
    """

    environment = DistillationSequenceEnvironment()
    state, mask = environment.reset()
    result = environment.step(2)

    assert state == INITIAL_STATE
    assert mask[:3] == (True, True, True)
    assert result.state == encode_state(("AB", "CD"))
    assert result.reward == pytest.approx(-1.654600)
    assert result.terminated is False
    assert result.info.selected_actions == (2,)
    assert result.info.selection_vector == (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    assert result.info.cumulative_cost_musd_per_year == pytest.approx(1.654600)


@pytest.mark.parametrize(
    ("actions", "expected_cost"),
    [
        ((2, 8, 10), 3.308330),
        ((2, 10, 8), 3.308330),
        ((1, 4, 8), 3.927360),
        ((3, 7, 10), 4.102530),
        ((1, 5, 9), 4.123155),
        ((3, 6, 9), 4.573980),
    ],
)
def test_complete_flowsheets_match_exact_costs(
    actions: tuple[int, ...], expected_cost: float
) -> None:
    """Verify exact cumulative cost for every feasible flowsheet structure.

    Inputs:
        actions: Complete feasible column sequence.
        expected_cost: Annual cost calculated from Table 17.1.
    Returns:
        None. The test passes when status, reward, and cost all match.
    """

    environment = DistillationSequenceEnvironment()
    total_reward = 0.0
    final_result = None

    # Three feasible sharp splits must complete a four-component separation sequence.
    for action_id in actions:
        final_result = environment.step(action_id)
        total_reward += final_result.reward

    assert final_result is not None
    assert final_result.terminated is True
    assert final_result.state == TERMINAL_STATE
    assert final_result.info.cumulative_cost_musd_per_year == pytest.approx(expected_cost)
    assert total_reward == pytest.approx(-expected_cost)
    assert set(final_result.info.selected_actions) == set(actions)


def test_environment_rejects_invalid_actions_without_advancing() -> None:
    """Verify that rejected actions do not advance the environment.

    Inputs:
        A new environment, unknown action 11, and infeasible action 8.
    Returns:
        None. The test passes when action 2 still executes from the initial state.
    """

    environment = DistillationSequenceEnvironment()
    with pytest.raises(ValueError, match="未知动作编号"):
        environment.step(11)

    with pytest.raises(ValueError, match="不可行"):
        environment.step(8)

    result = environment.step(2)
    assert result.state == encode_state(("AB", "CD"))
    assert result.info.selected_actions == (2,)


def test_environment_requires_reset_after_termination() -> None:
    """Verify that termination requires reset and reset clears history.

    Inputs:
        An environment after execution of the complete optimal flowsheet.
    Returns:
        None. The test passes when extra steps fail and reset restores the start.
    """

    environment = DistillationSequenceEnvironment()
    for action_id in (2, 8, 10):
        environment.step(action_id)

    with pytest.raises(RuntimeError, match="已终止"):
        environment.step(1)

    state, mask = environment.reset()
    result = environment.step(1)
    assert state == INITIAL_STATE
    assert mask == action_mask(INITIAL_STATE)
    assert result.info.selected_actions == (1,)
