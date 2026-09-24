"""
Last modified time: 2026-09-24
Last modified content: Consolidate verified chemistry and remove redundant work
Last modified by: OpenAI Codex
File design: Pure state functions with a stateful episode wrapper
File purpose: Calculate costs, legal actions, action masks, and sharp-split transitions
File creator: OpenAI Codex
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from itertools import product
import math

from distillation_sequencing_env.data import (
    ACTION_COUNT,
    COLUMN_BY_ACTION,
    COMPONENT_MOLE_FRACTIONS,
    INITIAL_STATE,
    INTERMEDIATE_MIXTURES,
    MIXTURE_ORDER,
    TERMINAL_STATE,
    TOTAL_FEED_KMOL_PER_HOUR,
    TOTAL_UTILITY_COST,
)
from distillation_sequencing_env.models import ActionMask, ColumnEconomics, State, StepInfo, StepResult


def _validate_state(state: State) -> None:
    """Validate that a state is a six-element binary tuple.

    Inputs:
        state: Environment state to validate.
    Returns:
        None. Raises ValueError when the state is invalid.
    """


    if type(state) is not tuple or len(state) != len(MIXTURE_ORDER):
        raise ValueError("State must be a six-element tuple.")
    if any(type(value) is not int or value not in (0, 1) for value in state):
        raise ValueError("State entries must be integer zero or one.")
    if state not in _STATE_DETAILS:
        raise ValueError("Active mixtures must not overlap in components.")


def encode_state(active: Iterable[str]) -> State:
    """Encode active mixed streams as a six-element state.

    Inputs:
        active: Names of mixed streams that still require separation.
    Returns:
        A binary state ordered by the fixed stream convention.
    """

    mixtures = tuple(active)
    unknown = set(mixtures) - INTERMEDIATE_MIXTURES
    if unknown:
        raise ValueError(f"未知混合物流: {sorted(unknown)}")

    if len(mixtures) != len(set(mixtures)):
        raise ValueError("活跃混合物流不能重复。")

    state = tuple(int(name in mixtures) for name in MIXTURE_ORDER)
    _validate_state(state)
    return state  # type: ignore[return-value]


def active_mixtures(state: State) -> tuple[str, ...]:
    """Decode a six-element state into active mixed streams.

    Inputs:
        state: Six-element binary environment state.
    Returns:
        Active mixed-stream names in the fixed stream order.
    """

    _validate_state(state)
    return _STATE_DETAILS[state][0]


def available_actions(state: State) -> tuple[int, ...]:
    """Return all textbook column IDs that are feasible in a state.

    Inputs:
        state: Six-element binary environment state.
    Returns:
        Feasible action IDs in ascending order.
    """

    _validate_state(state)
    return _STATE_DETAILS[state][1]


def action_mask(state: State) -> ActionMask:
    """Build a Boolean mask for textbook action IDs 1 through 10.

    Inputs:
        state: Six-element binary environment state.
    Returns:
        A length-10 mask whose index equals the action ID minus one.
    """

    _validate_state(state)
    return _STATE_DETAILS[state][2]


def _calculate_column_economics(action_id: int) -> ColumnEconomics:
    """Calculate flow, heat duty, and cost from the textbook equations.

    Inputs:
        action_id: Textbook column ID in the range 1 through 10.
    Returns:
        ColumnEconomics expressed with the textbook scaling conventions.
    """

    column = COLUMN_BY_ACTION[action_id]
    mole_fraction = sum(COMPONENT_MOLE_FRACTIONS[item] for item in column.feed_mixture)
    feed_flow = TOTAL_FEED_KMOL_PER_HOUR * mole_fraction
    heat_duty = column.heat_duty_coefficient * feed_flow
    annual_cost = (
        column.fixed_cost_kusd_per_year
        + column.variable_cost_factor * feed_flow
        + TOTAL_UTILITY_COST * heat_duty
    )
    return ColumnEconomics(feed_flow, heat_duty, annual_cost, annual_cost / 1000.0)


def apply_action(state: State, action_id: int) -> State:
    """Apply one feasible deterministic sharp split to a state.

    Inputs:
        state: Six-element state before the action.
        action_id: Candidate column ID from the textbook.
    Returns:
        The next state after replacing the feed with non-pure products.
    """

    valid = available_actions(state)
    if type(action_id) is not int or action_id not in COLUMN_BY_ACTION:
        raise ValueError(f"未知动作编号: {action_id}")

    if action_id not in valid:
        raise ValueError(f"动作 {action_id} 在当前状态不可行。")

    return _next_state(state, action_id)


@lru_cache(maxsize=12)
def _next_state(state: State, action_id: int) -> State:
    """Cache deterministic successors after public argument validation.

    Inputs:
        state: Validated reachable state.
        action_id: Feasible integer tower ID.
    Returns:
        Immutable successor state; at most twelve pairs are retained.
    """

    column = COLUMN_BY_ACTION[action_id]
    next_active = set(_STATE_DETAILS[state][0]) - {column.feed_mixture}
    next_active.update(product for product in (column.light_product, column.heavy_product)
                       if product in INTERMEDIATE_MIXTURES)
    return tuple(int(name in next_active) for name in MIXTURE_ORDER)


def _state_details() -> dict:
    """Precompute the eight valid partitions of the fixed textbook instance.

    Inputs:
        None. Reads immutable textbook stream and tower specifications.
    Returns:
        Mapping from state to active streams, feasible IDs, and Boolean mask.
    """

    details = {}
    # Reject overlapping streams; omitted components are already pure products.
    for state in product((0, 1), repeat=len(MIXTURE_ORDER)):
        active = tuple(name for name, bit in zip(MIXTURE_ORDER, state) if bit)
        components = "".join(active)
        if len(components) != len(set(components)):
            continue
        actions = tuple(key for key, column in COLUMN_BY_ACTION.items()
                        if column.feed_mixture in active)
        mask = tuple(key in actions for key in range(1, ACTION_COUNT + 1))
        details[state] = active, actions, mask
    return details


def calculate_column_economics(action_id: int) -> ColumnEconomics:
    """Return precomputed economics for an integer textbook tower ID.

    Inputs:
        action_id: Integer in the inclusive range one through ten.
    Returns:
        Immutable flow, duty, and annual cost record.
    """

    if type(action_id) is not int or action_id not in COLUMN_BY_ACTION:
        raise ValueError(f"未知动作编号: {action_id}")
    return _COLUMN_ECONOMICS[action_id]


def validate_reward(action_id: int, reward: float) -> None:
    """Require the observed undiscounted reward to match the tower cost.

    Inputs:
        action_id: Selected integer textbook tower ID.
        reward: Observed negative annual cost in millions of dollars per year.
    Returns:
        None. Raises ValueError for a nonfinite or inconsistent reward.
    """

    expected = -calculate_column_economics(action_id).annual_cost_musd_per_year
    if not math.isfinite(reward):
        raise ValueError("reward must be finite.")
    if abs(reward - expected) > 1e-9:
        raise ValueError("reward does not match the selected tower cost.")


_STATE_DETAILS = _state_details()
_COLUMN_ECONOMICS = {key: _calculate_column_economics(key) for key in COLUMN_BY_ACTION}



class DistillationSequenceEnvironment:
    """Manage one deterministic four-component separation episode.

    Inputs:
        None. The environment always starts from the textbook ABCD feed.
    Returns:
        States, rewards, masks, and engineering details through reset and step.
    """

    def __init__(self) -> None:
        """Initialize the environment state and episode records.

        Inputs:
            None.
        Returns:
            None. Creates the initial state, zero cost, and empty action history.
        """

        self._state = INITIAL_STATE
        self._cumulative_cost = 0.0
        self._selected_actions: list[int] = []

    def reset(self) -> tuple[State, ActionMask]:
        """Restore the environment to the unseparated ABCD feed.

        Inputs:
            None.
        Returns:
            The initial six-element state and its action mask.
        """

        self._state = INITIAL_STATE
        self._cumulative_cost = 0.0
        self._selected_actions = []
        return self._state, action_mask(self._state)

    def step(self, action_id: int) -> StepResult:
        """Execute one feasible column and return a complete step result.

        Inputs:
            action_id: Textbook column ID in the range 1 through 10.
        Returns:
            StepResult containing the next state, reward, status, and economics.
        """

        if self._state == TERMINAL_STATE:
            raise RuntimeError("episode 已终止；请先调用 reset。")

        if type(action_id) is not int or action_id not in COLUMN_BY_ACTION:
            raise ValueError(f"未知动作编号: {action_id}")

        column = COLUMN_BY_ACTION[action_id]
        economics = calculate_column_economics(action_id)
        self._state = apply_action(self._state, action_id)
        self._cumulative_cost += economics.annual_cost_musd_per_year
        self._selected_actions.append(action_id)
        selection_vector = tuple(
            int(candidate in self._selected_actions)
            for candidate in range(1, ACTION_COUNT + 1)
        )
        info = StepInfo(
            action_id,
            column.feed_mixture,
            economics,
            self._cumulative_cost,
            tuple(self._selected_actions),
            selection_vector,
        )
        return StepResult(
            self._state,
            -economics.annual_cost_musd_per_year,
            self._state == TERMINAL_STATE,
            action_mask(self._state),
            info,
        )
