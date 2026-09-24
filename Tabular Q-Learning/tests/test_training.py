"""
Last modified time: 2026-09-14-15:44
Last modified content: Test bounded training, epsilon scheduling, and evaluation
Last modified by: OpenAI Codex
File design: Unit and integration tests for episode orchestration
File purpose: Verify phase-three training and pure-strategy evaluation
File creator: OpenAI Codex
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

import distillation_q_learning.training as training_module
from distillation_q_learning import (
    INITIAL_STATE,
    TERMINAL_STATE,
    DistillationSequenceEnvironment,
    QLearningConfig,
    action_mask,
    encode_state,
    evaluate,
    train,
)


class EarlyTerminationEnvironment(DistillationSequenceEnvironment):
    """Simulate an invalid environment that terminates after one transition."""

    def step(self, action_id: int):
        """Return a terminal result before the required third transition.

        Inputs:
            action_id: Legal textbook action selected by the agent.
        Returns:
            Step result modified to report premature termination.
        """

        result = super().step(action_id)
        return replace(
            result,
            state=TERMINAL_STATE,
            terminated=True,
            action_mask=action_mask(TERMINAL_STATE),
        )


class MissingTerminationEnvironment(DistillationSequenceEnvironment):
    """Simulate an invalid environment that remains active after three transitions."""

    def step(self, action_id: int):
        """Replace a genuine terminal result with a nonterminal result.

        Inputs:
            action_id: Legal textbook action selected by the agent.
        Returns:
            Original result until the terminal transition, then an active CD state.
        """

        result = super().step(action_id)
        if not result.terminated:
            return result

        active_state = encode_state(("CD",))
        return replace(
            result,
            state=active_state,
            terminated=False,
            action_mask=action_mask(active_state),
        )


@pytest.fixture(scope="module")
def trained_agent():
    """Build the default accepted agent once for integration assertions.

    Inputs:
        None.
    Returns:
        Agent trained for the required 10,000 episodes with seed 42.
    """

    return train(QLearningConfig(seed=42), episodes=10_000)


@pytest.mark.parametrize("episodes", [0, -1])
def test_train_rejects_nonpositive_episode_counts(episodes: int) -> None:
    """Verify that training requires at least one episode.

    Inputs:
        episodes: Nonpositive episode count.
    Returns:
        None. The test passes when training raises ValueError.
    """

    with pytest.raises(ValueError, match="positive"):
        train(episodes=episodes)


@pytest.mark.parametrize("episodes", [1.5, True])
def test_train_rejects_noninteger_episode_counts(episodes: object) -> None:
    """Verify that training rejects noninteger episode counts and Boolean values.

    Inputs:
        episodes: Invalid episode-count value.
    Returns:
        None. The test passes when training raises TypeError.
    """

    with pytest.raises(TypeError, match="integer"):
        train(episodes=episodes)  # type: ignore[arg-type]


def test_train_uses_episode_indexed_epsilon_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify training obtains one scheduled epsilon value for every episode.

    Inputs:
        monkeypatch: Pytest replacement utility for observing schedule calls.
    Returns:
        None. The test passes when episode indices are supplied in order.
    """

    observed_episodes: list[int] = []

    def observe_schedule(config: QLearningConfig, episode: int) -> float:
        """Record an episode index and return deterministic greedy exploration.

        Inputs:
            config: Active Q-learning configuration.
            episode: Current zero-based episode index.
        Returns:
            Zero exploration probability.
        """

        assert config == QLearningConfig(seed=3)
        observed_episodes.append(episode)
        return 0.0

    monkeypatch.setattr(training_module, "epsilon_for_episode", observe_schedule)
    agent = train(QLearningConfig(seed=3), episodes=4)
    assert agent.config == QLearningConfig(seed=3)
    assert observed_episodes == [0, 1, 2, 3]


def test_default_training_recovers_required_initial_q_values(trained_agent) -> None:
    """Verify the accepted initial-state values after 10,000 episodes.

    Inputs:
        trained_agent: Module-scoped agent trained with the approved defaults.
    Returns:
        None. The test passes when all three textbook sequence costs are learned.
    """

    assert trained_agent.q_values(INITIAL_STATE) == pytest.approx(
        {1: -3.927360, 2: -3.308330, 3: -4.102530},
        abs=1e-6,
    )


def test_evaluate_returns_optimal_immutable_result(trained_agent) -> None:
    """Verify the optimal pure strategy, normalized flowsheet, and economics.

    Inputs:
        trained_agent: Agent that has converged under the approved experiment.
    Returns:
        None. The test passes when evaluation matches the textbook optimum.
    """

    result = evaluate(trained_agent)

    # The trajectory order is deterministic while the normalized set identifies the flowsheet.
    assert result.final_state == TERMINAL_STATE
    assert result.action_sequence == (2, 8, 10)
    assert result.normalized_column_set == (2, 8, 10)
    assert result.selection_vector == (0, 1, 0, 0, 0, 0, 0, 1, 0, 1)
    assert result.total_cost_musd_per_year == pytest.approx(3.308330)
    assert result.total_reward == pytest.approx(-3.308330)

    with pytest.raises(FrozenInstanceError):
        result.total_reward = 0.0  # type: ignore[misc]


def test_evaluate_preserves_q_values_and_random_state() -> None:
    """Verify that greedy evaluation neither learns nor consumes exploration randomness.

    Inputs:
        Two independently trained agents using the same configuration.
    Returns:
        None. The test passes when post-evaluation values and random choices match.
    """

    config = QLearningConfig(seed=19)
    evaluated_agent = train(config, episodes=500)
    control_agent = train(config, episodes=500)
    values_before = evaluated_agent.q_values(INITIAL_STATE)

    evaluate(evaluated_agent)

    assert evaluated_agent.q_values(INITIAL_STATE) == values_before
    assert evaluated_agent.select_action(INITIAL_STATE, 1.0) == control_agent.select_action(
        INITIAL_STATE, 1.0
    )


@pytest.mark.parametrize(
    ("environment_type", "message"),
    [
        (EarlyTerminationEnvironment, "before exactly three"),
        (MissingTerminationEnvironment, "after exactly three"),
    ],
)
def test_train_detects_invalid_episode_length(
    monkeypatch: pytest.MonkeyPatch,
    environment_type: type[DistillationSequenceEnvironment],
    message: str,
) -> None:
    """Verify training rejects both premature and missing termination.

    Inputs:
        monkeypatch: Pytest replacement utility.
        environment_type: Invalid environment behavior to inject.
        message: Expected runtime-error fragment.
    Returns:
        None. The test passes when the three-step invariant is enforced.
    """

    monkeypatch.setattr(training_module, "DistillationSequenceEnvironment", environment_type)
    with pytest.raises(RuntimeError, match=message):
        train(QLearningConfig(seed=1), episodes=1)


@pytest.mark.parametrize(
    ("environment_type", "message"),
    [
        (EarlyTerminationEnvironment, "before exactly three"),
        (MissingTerminationEnvironment, "after exactly three"),
    ],
)
def test_evaluate_detects_invalid_episode_length(
    monkeypatch: pytest.MonkeyPatch,
    environment_type: type[DistillationSequenceEnvironment],
    message: str,
) -> None:
    """Verify evaluation rejects both premature and missing termination.

    Inputs:
        monkeypatch: Pytest replacement utility.
        environment_type: Invalid environment behavior to inject.
        message: Expected runtime-error fragment.
    Returns:
        None. The test passes when evaluation enforces exactly three steps.
    """

    agent = train(QLearningConfig(seed=1), episodes=500)
    monkeypatch.setattr(training_module, "DistillationSequenceEnvironment", environment_type)
    with pytest.raises(RuntimeError, match=message):
        evaluate(agent)
