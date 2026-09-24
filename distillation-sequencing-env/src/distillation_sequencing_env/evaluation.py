"""
Last modified time: 2026-09-24
Last modified content: Share deterministic graph traversal and policy evaluation
Last modified by: OpenAI Codex
File design: Algorithm-independent evaluation helpers
File purpose: Report legal policies without changing learning or randomness
File creator: OpenAI Codex
"""

from collections import deque
from collections.abc import Callable

from distillation_sequencing_env.data import EPISODE_STEPS, INITIAL_STATE, TERMINAL_STATE
from distillation_sequencing_env.environment import (
    DistillationSequenceEnvironment, action_mask, active_mixtures, apply_action, available_actions,
)
from distillation_sequencing_env.models import ActionMask, EvaluationResult, State


def reachable_states(canonical: bool = False) -> tuple[State, ...]:
    """Enumerate decision states in discovery or canonical depth order.

    Inputs:
        canonical: Sort equal-depth states by their tuple when true.
    Returns:
        All seven nonterminal states, each appearing exactly once.
    """

    pending = deque((INITIAL_STATE,))
    depths = {INITIAL_STATE: 0}
    # Discover every nonterminal successor once using only legal transitions.
    while pending:
        state = pending.popleft()
        for action_id in available_actions(state):
            successor = apply_action(state, action_id)
            if successor != TERMINAL_STATE and successor not in depths:
                depths[successor] = depths[state] + 1
                pending.append(successor)
    if canonical:
        return tuple(sorted(depths, key=lambda state: (depths[state], state)))
    return tuple(depths)


def evaluate_greedy(
    choose: Callable[[State, ActionMask], int],
    environment: DistillationSequenceEnvironment,
) -> EvaluationResult:
    """Evaluate a deterministic callback over a complete legal flowsheet.

    Inputs:
        choose: Pure action selection callback receiving a state and its mask.
        environment: Fresh environment, also allowing test doubles for boundary checks.
    Returns:
        Ordered trajectory, canonical tower set, annual cost, and total reward.
    """

    state, mask = environment.reset()
    total_reward = 0.0
    # Advance exactly one complete horizon without calling a learning update.
    for index in range(EPISODE_STEPS):
        result = environment.step(choose(state, mask))
        total_reward += result.reward
        state, mask = result.state, result.action_mask
        if result.terminated and index != EPISODE_STEPS - 1:
            raise RuntimeError("Evaluation terminated before exactly three steps.")
    if not result.terminated or state != TERMINAL_STATE:
        raise RuntimeError("Evaluation did not terminate after exactly three steps.")
    return EvaluationResult(
        state, result.info.selected_actions, tuple(sorted(result.info.selected_actions)),
        result.info.selection_vector, result.info.cumulative_cost_musd_per_year, total_reward,
    )


def policy_records(
    inspect: Callable[[State, ActionMask], tuple[tuple[float, ...], int]],
) -> tuple[dict[str, object], ...]:
    """Serialize a probability policy using one inference per state.

    Inputs:
        inspect: Callback returning masked probabilities and a greedy tower ID.
    Returns:
        JSON-ready policy rows in breadth-first discovery order.
    """

    records = []
    # Keep probability and greedy records tied to the same network inference.
    for state in reachable_states():
        probabilities, greedy_action = inspect(state, action_mask(state))
        records.append({
            "state": state,
            "active_mixtures": active_mixtures(state),
            "legal_actions": available_actions(state),
            "probabilities": probabilities,
            "greedy_action": greedy_action,
        })
    return tuple(records)
