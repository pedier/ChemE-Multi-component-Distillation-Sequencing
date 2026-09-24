"""
Last modified time: 2026-09-21-03:13
Last modified content: Test DQN episode scheduling, exact length, and pure evaluation
Last modified by: OpenAI Codex
File design: Training and evaluation unit and integration tests
File purpose: Verify online-only learning and the fixed-seed textbook optimum
File creator: OpenAI Codex
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

import distillation_dqn.training as training_module
from distillation_dqn import (
    DQNConfig,
    DuelingDoubleDQNAgent,
    DistillationSequenceEnvironment,
    INITIAL_STATE,
    TERMINAL_STATE,
    action_mask,
    encode_state,
    epsilon_for_episode,
    evaluate,
    train,
)


class EarlyTerminationEnvironment(DistillationSequenceEnvironment):
    """Simulate an invalid environment that terminates after one split.

    Inputs:
        Legal tower IDs selected by a DQN agent.
    Returns:
        Step records whose first transition falsely reports completion.
    """

    def step(self, action_id: int):
        """Replace the first legal step with premature termination.

        Inputs:
            action_id: Selected textbook tower ID.
        Returns:
            Step result with a terminal state and no next actions.
        """

        result = super().step(action_id)
        return replace(result, state=TERMINAL_STATE, terminated=True,
                       action_mask=action_mask(TERMINAL_STATE))


class MissingTerminationEnvironment(DistillationSequenceEnvironment):
    """Simulate an invalid environment active after three splits.

    Inputs:
        Legal tower IDs selected by a DQN agent.
    Returns:
        A nonterminal result in place of genuine completion.
    """

    def step(self, action_id: int):
        """Replace a genuine terminal result with an active CD stream.

        Inputs:
            action_id: Selected textbook tower ID.
        Returns:
            Original result before termination, then a false active result.
        """

        result = super().step(action_id)
        if not result.terminated:
            return result
        active_state = encode_state(("CD",))
        return replace(result, state=active_state, terminated=False,
                       action_mask=action_mask(active_state))


@pytest.fixture(scope="module")
def trained_agent() -> DuelingDoubleDQNAgent:
    """Train the accepted default experiment once for integration tests.

    Inputs:
        None.
    Returns:
        Agent trained for 10,000 episodes with seed 42.
    """

    return train(DQNConfig(seed=42), episodes=10_000)


def test_epsilon_schedule_decays_and_clips() -> None:
    """Verify per-episode exponential exploration with a fixed floor.

    Inputs:
        Early and late zero-based episode indices.
    Returns:
        None. The test passes when the agreed schedule is reproduced.
    """

    assert epsilon_for_episode(0) == 1.0
    assert epsilon_for_episode(1) == pytest.approx(0.995)
    assert epsilon_for_episode(10_000) == 0.05


@pytest.mark.parametrize(("episode", "exception"), [(-1, ValueError), (1.5, TypeError),
                                                       (True, TypeError)])
def test_epsilon_schedule_rejects_invalid_indices(
    episode: object, exception: type[Exception]
) -> None:
    """Reject negative or noninteger episode indices.

    Inputs:
        episode: Invalid index.
        exception: Expected validation exception.
    Returns:
        None. The test passes when scheduling fails clearly.
    """

    with pytest.raises(exception):
        epsilon_for_episode(episode)  # type: ignore[arg-type]


@pytest.mark.parametrize(("episodes", "exception"), [(0, ValueError), (-1, ValueError),
                                                        (1.5, TypeError), (True, TypeError)])
def test_train_rejects_invalid_episode_counts(
    episodes: object, exception: type[Exception]
) -> None:
    """Require a positive integral number of independent episodes.

    Inputs:
        episodes: Invalid requested episode count.
        exception: Expected exception category.
    Returns:
        None. The test passes when training rejects the count.
    """

    with pytest.raises(exception):
        train(episodes=episodes)  # type: ignore[arg-type]


def test_train_uses_one_epsilon_per_episode_and_three_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify episode scheduling and exact three-step replay collection.

    Inputs:
        monkeypatch: Replaces the schedule to record episode indices.
    Returns:
        None. The test passes when four episodes store twelve transitions.
    """

    observed: list[int] = []

    def record_schedule(episode: int) -> float:
        """Record a schedule query and force pure greedy selection.

        Inputs:
            episode: Zero-based episode index.
        Returns:
            Zero exploration probability.
        """

        observed.append(episode)
        return 0.0

    monkeypatch.setattr(training_module, "epsilon_for_episode", record_schedule)
    config = DQNConfig(replay_capacity=100, batch_size=4, warmup_steps=8, seed=4)
    agent = train(config, episodes=4)
    assert agent.config == config
    assert observed == [0, 1, 2, 3]
    assert len(agent.replay) == 12
    assert agent.optimization_steps == 5


def test_same_seed_produces_same_trained_values() -> None:
    """Verify short-run reproducibility after neural optimization begins.

    Inputs:
        Two independent 80-episode runs with the same small-batch config.
    Returns:
        None. The test passes when weights and feasible Q values match.
    """

    config = DQNConfig(replay_capacity=100, batch_size=4, warmup_steps=8,
                       target_sync_interval=5, seed=19)
    first = train(config, episodes=80)
    second = train(config, episodes=80)
    assert first.q_values(INITIAL_STATE) == second.q_values(INITIAL_STATE)
    for name, weights in first.online_network.state_dict().items():
        assert torch.equal(weights, second.online_network.state_dict()[name])


@pytest.mark.parametrize(
    ("environment_type", "message"),
    [(EarlyTerminationEnvironment, "before exactly three"),
     (MissingTerminationEnvironment, "after exactly three")],
)
def test_train_rejects_invalid_episode_length(
    monkeypatch: pytest.MonkeyPatch,
    environment_type: type[DistillationSequenceEnvironment],
    message: str,
) -> None:
    """Detect both premature and missing process completion during training.

    Inputs:
        monkeypatch: Injects an invalid environment.
        environment_type: Environment violating the three-split invariant.
        message: Expected runtime-error fragment.
    Returns:
        None. The test passes when training fails before storing bad data.
    """

    monkeypatch.setattr(training_module, "DistillationSequenceEnvironment", environment_type)
    with pytest.raises(RuntimeError, match=message):
        train(DQNConfig(seed=1), episodes=1)


def test_default_training_finds_textbook_optimum(trained_agent: DuelingDoubleDQNAgent) -> None:
    """Verify fixed-seed greedy evaluation of the complete default run.

    Inputs:
        trained_agent: Agent from 10,000 online episodes with seed 42.
    Returns:
        None. The test passes for the optimal tower set and economics.
    """

    result = evaluate(trained_agent)
    assert result.final_state == TERMINAL_STATE
    assert result.normalized_column_set == (2, 8, 10)
    assert result.action_sequence in ((2, 8, 10), (2, 10, 8))
    assert result.selection_vector == (0, 1, 0, 0, 0, 0, 0, 1, 0, 1)
    assert result.total_cost_musd_per_year == pytest.approx(3.308330)
    assert result.total_reward == pytest.approx(-3.308330)
    with pytest.raises(FrozenInstanceError):
        result.total_reward = 0.0  # type: ignore[misc]


def test_evaluate_does_not_explore_or_update(trained_agent: DuelingDoubleDQNAgent) -> None:
    """Verify pure evaluation preserves weights, replay, and random state.

    Inputs:
        trained_agent: Default fully trained DQN agent.
    Returns:
        None. The test passes when evaluation has no learning side effects.
    """

    random_state = trained_agent._random.getstate()
    weights = {name: value.clone() for name, value in
               trained_agent.online_network.state_dict().items()}
    updates = trained_agent.optimization_steps
    replay_size = len(trained_agent.replay)

    evaluate(trained_agent)
    assert trained_agent._random.getstate() == random_state
    assert trained_agent.optimization_steps == updates
    assert len(trained_agent.replay) == replay_size
    for name, value in weights.items():
        assert torch.equal(value, trained_agent.online_network.state_dict()[name])


@pytest.mark.parametrize(
    ("environment_type", "message"),
    [(EarlyTerminationEnvironment, "before exactly three"),
     (MissingTerminationEnvironment, "after exactly three")],
)
def test_evaluate_rejects_invalid_episode_length(
    monkeypatch: pytest.MonkeyPatch,
    environment_type: type[DistillationSequenceEnvironment],
    message: str,
) -> None:
    """Detect invalid completion length during greedy evaluation.

    Inputs:
        monkeypatch: Injects one invalid chemical environment.
        environment_type: Premature or missing completion behavior.
        message: Expected runtime-error fragment.
    Returns:
        None. The test passes when evaluation refuses an invalid flowsheet.
    """

    agent = DuelingDoubleDQNAgent(DQNConfig(seed=1))
    monkeypatch.setattr(training_module, "DistillationSequenceEnvironment", environment_type)
    with pytest.raises(RuntimeError, match=message):
        evaluate(agent)
