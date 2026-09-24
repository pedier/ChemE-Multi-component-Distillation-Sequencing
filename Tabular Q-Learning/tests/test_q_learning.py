"""
Last modified time: 2026-09-24
Last modified content: Verify shared contracts, strict inputs and seeded cleanup behavior
Last modified by: OpenAI Codex
File design: Unit tests for sparse Q values, exploration, policies, and Bellman updates
File purpose: Verify every public Q-learning behavior without a training loop
File creator: OpenAI Codex
"""

from __future__ import annotations

import pytest

from distillation_q_learning import (
    INITIAL_STATE,
    TERMINAL_STATE,
    QLearningConfig,
    TabularQLearningAgent,
    encode_state,
    epsilon_for_episode,
)


def test_q_learning_config_defaults_preserve_economic_objective() -> None:
    """Verify the documented stage-two default hyperparameters.

    Inputs:
        A default QLearningConfig.
    Returns:
        None. The test passes when all defaults match the approved plan.
    """

    config = QLearningConfig()
    assert config.learning_rate == 0.2
    assert config.discount_factor == 1.0
    assert config.epsilon_start == 1.0
    assert config.epsilon_min == 0.05
    assert config.epsilon_decay == 0.995
    assert config.seed == 42


@pytest.mark.parametrize(
    ("overrides", "exception"),
    [
        ({"learning_rate": 0.0}, ValueError),
        ({"learning_rate": 1.1}, ValueError),
        ({"discount_factor": 0.99}, ValueError),
        ({"epsilon_min": -0.1}, ValueError),
        ({"epsilon_start": 1.1}, ValueError),
        ({"epsilon_min": 0.6, "epsilon_start": 0.5}, ValueError),
        ({"epsilon_decay": 0.0}, ValueError),
        ({"epsilon_decay": 1.1}, ValueError),
        ({"seed": 1.5}, TypeError),
        ({"seed": True}, TypeError),
    ],
)
def test_q_learning_config_rejects_invalid_values(
    overrides: dict[str, float], exception: type[Exception]
) -> None:
    """Verify validation for every Q-learning configuration field.

    Inputs:
        overrides: One invalid configuration override.
        exception: Expected exception type.
    Returns:
        None. The test passes when construction raises the expected exception.
    """

    with pytest.raises(exception):
        QLearningConfig(**overrides)  # type: ignore[arg-type]


def test_epsilon_for_episode_decays_and_clips_at_minimum() -> None:
    """Verify exponential epsilon decay and lower-bound clipping.

    Inputs:
        A configuration with simple decimal decay.
    Returns:
        None. The test passes when early and late episode values match.
    """

    config = QLearningConfig(epsilon_start=0.8, epsilon_min=0.2, epsilon_decay=0.5)
    assert epsilon_for_episode(config, 0) == pytest.approx(0.8)
    assert epsilon_for_episode(config, 1) == pytest.approx(0.4)
    assert epsilon_for_episode(config, 2) == pytest.approx(0.2)
    assert epsilon_for_episode(config, 20) == pytest.approx(0.2)


def test_epsilon_for_episode_rejects_negative_index() -> None:
    """Verify that an episode index cannot be negative.

    Inputs:
        Default configuration and episode index negative one.
    Returns:
        None. The test passes when epsilon calculation raises ValueError.
    """

    with pytest.raises(ValueError, match="non-negative"):
        epsilon_for_episode(QLearningConfig(), -1)


def test_agent_initializes_sparse_zero_values_for_legal_actions() -> None:
    """Verify empty-table initialization and legal-action filtering.

    Inputs:
        A new default agent and the initial distillation state.
    Returns:
        None. The test passes when only actions 1, 2, and 3 read as zero.
    """

    agent = TabularQLearningAgent()
    values = agent.q_values(INITIAL_STATE)
    assert values == {1: 0.0, 2: 0.0, 3: 0.0}
    assert agent.q_value(INITIAL_STATE, 2) == 0.0

    values[1] = 99.0
    assert agent.q_value(INITIAL_STATE, 1) == 0.0


@pytest.mark.parametrize(
    ("state", "action_id"),
    [(INITIAL_STATE, 8), (TERMINAL_STATE, 1), (INITIAL_STATE, True), (INITIAL_STATE, 1.0)],
)
def test_q_value_rejects_infeasible_actions(state: tuple[int, ...], action_id: int) -> None:
    """Verify that illegal state-action pairs never enter the Q table.

    Inputs:
        state: Initial or terminal distillation state.
        action_id: Action that is infeasible in the supplied state.
    Returns:
        None. The test passes when Q-value access raises ValueError.
    """

    with pytest.raises(ValueError, match="infeasible"):
        TabularQLearningAgent().q_value(state, action_id)


def test_greedy_action_uses_smallest_id_for_exact_and_close_ties() -> None:
    """Verify deterministic pure-action selection for tied Q values.

    Inputs:
        An agent and the state containing active AB and CD streams.
    Returns:
        None. The test passes when action 8 wins a numerical tie with action 10.
    """

    config = QLearningConfig(learning_rate=1.0)
    agent = TabularQLearningAgent(config)
    state = encode_state(("AB", "CD"))
    agent.update(state, 8, -1.0, TERMINAL_STATE, True)
    agent.update(state, 10, -1.0 + 5e-13, TERMINAL_STATE, True)
    assert agent.greedy_action(state) == 8


def test_greedy_action_rejects_terminal_state() -> None:
    """Verify that a pure action cannot be extracted after termination.

    Inputs:
        A new agent and the terminal distillation state.
    Returns:
        None. The test passes when greedy selection raises ValueError.
    """

    with pytest.raises(ValueError, match="terminal"):
        TabularQLearningAgent().greedy_action(TERMINAL_STATE)


def test_select_action_with_zero_epsilon_is_greedy() -> None:
    """Verify pure greedy selection when exploration is disabled.

    Inputs:
        A new agent, initial state, and epsilon zero.
    Returns:
        None. The test passes when deterministic tie handling selects action 1.
    """

    agent = TabularQLearningAgent()
    assert agent.select_action(INITIAL_STATE, epsilon=0.0) == 1


def test_select_action_with_full_epsilon_is_masked_and_reproducible() -> None:
    """Verify seeded uniform exploration over legal actions only.

    Inputs:
        Two agents with the same seed, initial state, and epsilon one.
    Returns:
        None. The test passes when sampled sequences match and contain only legal IDs.
    """

    first = TabularQLearningAgent(QLearningConfig(seed=7))
    second = TabularQLearningAgent(QLearningConfig(seed=7))
    first_actions = [first.select_action(INITIAL_STATE, 1.0) for _ in range(30)]
    second_actions = [second.select_action(INITIAL_STATE, 1.0) for _ in range(30)]
    assert first_actions == second_actions
    assert set(first_actions) <= {1, 2, 3}
    assert len(set(first_actions)) > 1


@pytest.mark.parametrize("epsilon", [-0.1, 1.1])
def test_select_action_rejects_invalid_epsilon(epsilon: float) -> None:
    """Verify that action selection rejects an invalid exploration probability.

    Inputs:
        epsilon: Exploration value outside the closed unit interval.
    Returns:
        None. The test passes when action selection raises ValueError.
    """

    with pytest.raises(ValueError, match="epsilon"):
        TabularQLearningAgent().select_action(INITIAL_STATE, epsilon)


def test_select_action_rejects_terminal_state() -> None:
    """Verify that epsilon-greedy selection rejects the terminal state.

    Inputs:
        A new agent, terminal state, and a valid epsilon.
    Returns:
        None. The test passes when action selection raises ValueError.
    """

    with pytest.raises(ValueError, match="terminal"):
        TabularQLearningAgent().select_action(TERMINAL_STATE, 0.5)


def test_terminal_update_uses_reward_without_bootstrap() -> None:
    """Verify the Bellman update for a transition that ends the episode.

    Inputs:
        Column 8 transition from active CD to the terminal state.
    Returns:
        None. The test passes when the update moves 20 percent toward the reward.
    """

    agent = TabularQLearningAgent()
    state = encode_state(("CD",))
    updated = agent.update(state, 8, -1.016760, TERMINAL_STATE, True)
    assert updated == pytest.approx(-0.203352)
    assert agent.q_value(state, 8) == pytest.approx(updated)


def test_nonterminal_update_bootstraps_from_best_legal_next_action() -> None:
    """Verify masked Bellman bootstrap from the highest legal next Q value.

    Inputs:
        Initial action 2 followed by a synthetic AB-and-CD value table.
    Returns:
        None. The test passes when the target uses action 8 rather than action 10.
    """

    agent = TabularQLearningAgent(QLearningConfig(learning_rate=1.0))
    next_state = encode_state(("AB", "CD"))
    agent.update(next_state, 8, -1.0, TERMINAL_STATE, True)
    agent.update(next_state, 10, -2.0, TERMINAL_STATE, True)
    updated = agent.update(INITIAL_STATE, 2, -1.654600, next_state, False)
    assert updated == pytest.approx(-2.654600)


@pytest.mark.parametrize(
    ("next_state", "terminated", "message"),
    [
        (encode_state(("CD",)), True, "terminal transition"),
        (TERMINAL_STATE, False, "nonterminal transition"),
    ],
)
def test_update_rejects_inconsistent_termination_flags(
    next_state: tuple[int, ...], terminated: bool, message: str
) -> None:
    """Verify consistency between termination flags and next-state action sets.

    Inputs:
        next_state: State that contradicts the supplied termination flag.
        terminated: Incorrect termination flag.
        message: Expected validation message fragment.
    Returns:
        None. The test passes when the update raises ValueError.
    """

    agent = TabularQLearningAgent()
    with pytest.raises(ValueError, match=message):
        agent.update(INITIAL_STATE, 1, -1.0, next_state, terminated)


def test_greedy_policy_maps_each_supplied_nonterminal_state() -> None:
    """Verify deterministic policy extraction for multiple states.

    Inputs:
        A new agent and two nonterminal states with zero-valued ties.
    Returns:
        None. The test passes when each state maps to its smallest legal action.
    """

    agent = TabularQLearningAgent()
    bcd_state = encode_state(("BCD",))
    policy = agent.greedy_policy((INITIAL_STATE, bcd_state))
    assert policy == {INITIAL_STATE: 1, bcd_state: 4}

    with pytest.raises(ValueError, match="terminal"):
        agent.greedy_policy((TERMINAL_STATE,))
