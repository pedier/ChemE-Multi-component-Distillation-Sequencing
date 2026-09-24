"""
Last modified time: 2026-09-21-02:24
Last modified content: Test seeded masked agent selection and optimizer lifecycle
Last modified by: OpenAI Codex
File design: DQN configuration and agent unit tests
File purpose: Verify legal actions, replay warmup, gradient updates, and target sync
File creator: OpenAI Codex
"""

from __future__ import annotations

import pytest
import torch

from distillation_dqn import (
    DQNConfig,
    DuelingDoubleDQNAgent,
    DistillationSequenceEnvironment,
    INITIAL_STATE,
    TERMINAL_STATE,
    Transition,
    action_mask,
    encode_state,
)


def test_config_defaults_preserve_undiscounted_objective() -> None:
    """Check every agreed stage-two default value.

    Inputs:
        A default DQN configuration.
    Returns:
        None. The test passes when all fixed-instance defaults match the plan.
    """

    config = DQNConfig()
    assert config.learning_rate == 1e-3
    assert config.discount_factor == 1.0
    assert config.replay_capacity == 10_000
    assert config.batch_size == 64
    assert config.warmup_steps == 128
    assert config.target_sync_interval == 100
    assert config.gradient_clip_norm == 10.0
    assert config.seed == 42


@pytest.mark.parametrize(
    ("overrides", "exception"),
    [
        ({"learning_rate": 0.0}, ValueError),
        ({"learning_rate": float("inf")}, ValueError),
        ({"discount_factor": 0.99}, ValueError),
        ({"gradient_clip_norm": -1.0}, ValueError),
        ({"gradient_clip_norm": float("nan")}, ValueError),
        ({"replay_capacity": 0}, ValueError),
        ({"batch_size": 0}, ValueError),
        ({"warmup_steps": 1}, ValueError),
        ({"target_sync_interval": 0}, ValueError),
        ({"batch_size": 1.5}, TypeError),
        ({"seed": True}, TypeError),
    ],
)
def test_config_rejects_invalid_values(
    overrides: dict[str, object], exception: type[Exception]
) -> None:
    """Reject unstable or physically inconsistent DQN configuration.

    Inputs:
        overrides: One invalid parameter replacement.
        exception: Expected exception category.
    Returns:
        None. The test passes when validation raises that category.
    """

    with pytest.raises(exception):
        DQNConfig(**overrides)  # type: ignore[arg-type]


def test_agent_initializes_equal_frozen_targets_without_consuming_global_rng() -> None:
    """Verify seeded online weights, copied target weights, and RNG isolation.

    Inputs:
        Two agents using the same seed and the current PyTorch RNG state.
    Returns:
        None. The test passes when initialization is reproducible and isolated.
    """

    before = torch.get_rng_state().clone()
    first = DuelingDoubleDQNAgent(DQNConfig(seed=7))
    second = DuelingDoubleDQNAgent(DQNConfig(seed=7))
    assert torch.equal(before, torch.get_rng_state())

    for name, weights in first.online_network.state_dict().items():
        assert torch.equal(weights, second.online_network.state_dict()[name])
        assert torch.equal(weights, first.target_network.state_dict()[name])
    assert not any(parameter.requires_grad for parameter in first.target_network.parameters())


def test_agent_q_values_and_greedy_action_only_include_legal_towers() -> None:
    """Verify legal Q queries and stable smallest-ID tie behavior.

    Inputs:
        Agent with all network parameters zeroed.
    Returns:
        None. The test passes when illegal towers never appear or win ties.
    """

    agent = DuelingDoubleDQNAgent()
    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.zero_()

    assert agent.q_values(INITIAL_STATE) == {1: 0.0, 2: 0.0, 3: 0.0}
    assert agent.q_values(TERMINAL_STATE) == {}
    assert agent.greedy_action(INITIAL_STATE) == 1
    assert agent.greedy_action(encode_state(("AB", "CD"))) == 8
    with pytest.raises(ValueError, match="terminal"):
        agent.greedy_action(TERMINAL_STATE)


def test_agent_exploration_is_seeded_and_masked() -> None:
    """Sample only current legal actions with an independent seed.

    Inputs:
        Two equal-seed agents and the initial three-action mask.
    Returns:
        None. The test passes when action sequences match and stay feasible.
    """

    first = DuelingDoubleDQNAgent(DQNConfig(seed=11))
    second = DuelingDoubleDQNAgent(DQNConfig(seed=11))
    mask = action_mask(INITIAL_STATE)
    first_actions = [first.select_action(INITIAL_STATE, mask, 1.0) for _ in range(30)]
    second_actions = [second.select_action(INITIAL_STATE, mask, 1.0) for _ in range(30)]
    assert first_actions == second_actions
    assert set(first_actions) == {1, 2, 3}
    assert first.select_action(INITIAL_STATE, mask, 0.0) == first.greedy_action(INITIAL_STATE)


def test_agent_select_action_rejects_bad_mask_epsilon_and_terminal_state() -> None:
    """Reject an altered mask, invalid probability, or completed episode.

    Inputs:
        Agent with malformed selection requests.
    Returns:
        None. The test passes when each invalid request raises ValueError.
    """

    agent = DuelingDoubleDQNAgent()
    with pytest.raises(ValueError, match="mask"):
        agent.select_action(INITIAL_STATE, action_mask(TERMINAL_STATE), 0.1)
    for epsilon in (-0.1, 1.1):
        with pytest.raises(ValueError, match="epsilon"):
            agent.select_action(INITIAL_STATE, action_mask(INITIAL_STATE), epsilon)
    with pytest.raises(ValueError, match="terminal"):
        agent.select_action(TERMINAL_STATE, action_mask(TERMINAL_STATE), 0.5)


def test_observe_warms_up_updates_online_then_synchronizes_target() -> None:
    """Exercise replay storage, Huber optimization, and hard target sync.

    Inputs:
        Three actual consecutive environment transitions.
    Returns:
        None. The test passes when warmup and sync timing match the config.
    """

    config = DQNConfig(replay_capacity=4, batch_size=2, warmup_steps=2,
                       target_sync_interval=2, seed=3)
    agent = DuelingDoubleDQNAgent(config)
    environment = DistillationSequenceEnvironment()
    state, _ = environment.reset()
    initial_weights = agent.online_network.value_head.weight.detach().clone()

    for step_index, action_id in enumerate((2, 8, 10)):
        result = environment.step(action_id)
        record = Transition(state, action_id, result.reward, result.state,
                            result.action_mask, result.terminated)
        loss = agent.observe(record)
        if step_index == 0:
            assert loss is None
            assert agent.optimization_steps == 0
        else:
            assert loss is not None and loss >= 0.0
        if step_index == 1:
            assert not torch.equal(
                agent.online_network.value_head.bias, agent.target_network.value_head.bias
            )
        state = result.state

    assert len(agent.replay) == 3
    assert agent.optimization_steps == 2
    assert not torch.equal(initial_weights, agent.online_network.value_head.weight)
    for name, weights in agent.online_network.state_dict().items():
        assert torch.equal(weights, agent.target_network.state_dict()[name])


def test_manual_target_sync_copies_online_parameters() -> None:
    """Verify exact hard synchronization independent of the update interval.

    Inputs:
        Agent with an intentionally changed online value bias.
    Returns:
        None. The test passes when sync_target copies the changed weight.
    """

    agent = DuelingDoubleDQNAgent()
    with torch.no_grad():
        agent.online_network.value_head.bias.add_(1.0)
    assert not torch.equal(agent.online_network.value_head.bias, agent.target_network.value_head.bias)
    agent.sync_target()
    assert torch.equal(agent.online_network.value_head.bias, agent.target_network.value_head.bias)

