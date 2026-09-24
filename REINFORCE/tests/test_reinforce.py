"""
Last modified time: 2026-09-24
Last modified content: Verify shared contracts, strict inputs and seeded cleanup behavior
Last modified by: OpenAI Codex
File design: Focused episodic policy-gradient unit tests
File purpose: Verify baseline-free on-policy loss and deterministic evaluation choices
File creator: OpenAI Codex
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from distillation_reinforce import (
    INITIAL_STATE,
    DistillationSequenceEnvironment,
    EpisodeTrajectory,
    ReinforceAgent,
    ReinforceConfig,
    TrajectoryStep,
    action_mask,
    encode_state,
    masked_distribution,
    reward_to_go,
)
from distillation_reinforce.reinforce import _validate_trajectory


def test_reinforce_config_defaults_and_trajectory_records_are_immutable() -> None:
    """Check the fixed optimizer defaults and immutable episode records.

    Inputs:
        Default configuration and a sample trajectory step.
    Returns:
        None. Values match the design and frozen fields reject assignment.
    """

    config = ReinforceConfig()
    step = TrajectoryStep(INITIAL_STATE, 1, -1.0, torch.tensor(-0.5))
    episode = EpisodeTrajectory((step,), False)

    assert config.learning_rate == pytest.approx(1e-3)
    assert config.batch_episodes == 32
    assert config.seed == 42

    with pytest.raises(FrozenInstanceError):
        episode.terminated = True  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"learning_rate": 0.0},
        {"learning_rate": float("nan")},
        {"batch_episodes": 0},
        {"batch_episodes": 1.5},
        {"seed": True},
    ],
)
def test_reinforce_config_rejects_invalid_values(overrides: dict[str, object]) -> None:
    """Check optimizer settings cannot silently change the training contract.

    Inputs:
        overrides: One invalid learning rate, batch size, or seed.
    Returns:
        None. Construction raises ValueError.
    """

    with pytest.raises(ValueError):
        ReinforceConfig(**overrides)  # type: ignore[arg-type]


def test_reward_to_go_uses_undiscounted_suffix_sums() -> None:
    """Check that later tower costs retain their full economic weight.

    Inputs:
        The three negative incremental costs of the optimal flowsheet.
    Returns:
        None. Each return equals the corresponding reward suffix sum.
    """

    returns = reward_to_go((-1.654600, -1.016760, -0.636970))
    assert returns == pytest.approx((-3.308330, -1.653730, -0.636970))


@pytest.mark.parametrize("rewards", [(), (1.0, float("inf")), (float("nan"),)])
def test_reward_to_go_rejects_empty_or_nonfinite_rewards(rewards: tuple[float, ...]) -> None:
    """Check that invalid Monte Carlo returns cannot enter an update.

    Inputs:
        rewards: Empty or nonfinite incremental rewards.
    Returns:
        None. The return calculation raises ValueError.
    """

    with pytest.raises(ValueError):
        reward_to_go(rewards)


def test_agent_initialization_and_sampling_reproduce_with_seed() -> None:
    """Check seeded network weights and categorical action samples.

    Inputs:
        Two agents created sequentially from the same seed.
    Returns:
        None. Both produce identical starting weights and action sequences.
    """

    config = ReinforceConfig(seed=19)
    first = ReinforceAgent(config)
    initial_weights = [parameter.detach().clone() for parameter in first.policy.parameters()]
    first_actions = [first.sample_action(INITIAL_STATE, action_mask(INITIAL_STATE))[0] for _ in range(12)]

    second = ReinforceAgent(config)
    second_actions = [second.sample_action(INITIAL_STATE, action_mask(INITIAL_STATE))[0] for _ in range(12)]

    assert first_actions == second_actions
    assert all(torch.equal(a, b) for a, b in zip(initial_weights, second.policy.parameters()))


def test_agent_sample_action_returns_legal_id_and_trainable_log_probability() -> None:
    """Check external tower numbering and policy-gradient sample data.

    Inputs:
        Initial state with towers one through three legal.
    Returns:
        None. Every sampled ID is legal and its log probability has gradients.
    """

    agent = ReinforceAgent()
    mask = action_mask(INITIAL_STATE)

    for _ in range(20):
        action_id, log_probability = agent.sample_action(INITIAL_STATE, mask)
        assert action_id in (1, 2, 3)
        assert log_probability.ndim == 0
        assert log_probability.requires_grad


def test_agent_greedy_action_breaks_ties_without_random_sampling() -> None:
    """Check pure-strategy selection and non-mutating policy inspection.

    Inputs:
        AB plus CD state and a policy with all network parameters zero.
    Returns:
        None. Tower eight wins the exact tie and probabilities remain masked.
    """

    agent = ReinforceAgent()
    state = encode_state(("AB", "CD"))
    mask = action_mask(state)
    with torch.no_grad():
        for parameter in agent.policy.parameters():
            parameter.zero_()

    random_state = torch.get_rng_state().clone()
    probabilities = agent.action_probabilities(state, mask)
    action_id = agent.greedy_action(state, mask)

    assert action_id == 8
    assert probabilities[7] == pytest.approx(0.5)
    assert probabilities[9] == pytest.approx(0.5)
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(probabilities[index] == 0.0 for index in range(10) if not mask[index])
    assert torch.equal(random_state, torch.get_rng_state())


def test_complete_trajectory_validation_accepts_legal_three_step_path() -> None:
    """Check that the trajectory contract accepts a complete sharp-split path.

    Inputs:
        States and actions for towers one, four, and eight.
    Returns:
        None. Validation succeeds without modifying the episode.
    """

    log_probability = torch.tensor(-0.5, requires_grad=True)
    steps = (
        TrajectoryStep(INITIAL_STATE, 1, -1.553400, log_probability),
        TrajectoryStep(encode_state(("BCD",)), 4, -1.357200, log_probability),
        TrajectoryStep(encode_state(("CD",)), 8, -1.016760, log_probability),
    )
    assert _validate_trajectory(EpisodeTrajectory(steps, True)) is None


def test_trajectory_validation_rejects_incomplete_or_bad_samples() -> None:
    """Check that incomplete, illegal, and detached episodes cannot be updated.

    Inputs:
        Variants of an otherwise legal three-step sampled trajectory.
    Returns:
        None. Every invalid variant raises ValueError.
    """

    log_probability = torch.tensor(-0.5, requires_grad=True)
    steps = (
        TrajectoryStep(INITIAL_STATE, 1, -1.553400, log_probability),
        TrajectoryStep(encode_state(("BCD",)), 4, -1.357200, log_probability),
        TrajectoryStep(encode_state(("CD",)), 8, -1.016760, log_probability),
    )
    invalid = (
        EpisodeTrajectory(steps, False),
        EpisodeTrajectory(steps[:2], True),
        EpisodeTrajectory((replace(steps[0], action_id=8), *steps[1:]), True),
        EpisodeTrajectory((replace(steps[0], reward=float("inf")), *steps[1:]), True),
        EpisodeTrajectory((replace(steps[0], reward=0.0), *steps[1:]), True),
        EpisodeTrajectory((replace(steps[0], log_probability=torch.tensor(-0.5)), *steps[1:]), True),
    )

    for episode in invalid:
        with pytest.raises(ValueError):
            _validate_trajectory(episode)


def test_agent_update_matches_reward_to_go_loss_and_changes_weights() -> None:
    """Check one complete on-policy likelihood-ratio update.

    Inputs:
        Sampled-probability tensors for the feasible tower path one, four, eight.
    Returns:
        None. Reported loss matches the equation and policy weights change.
    """

    agent = ReinforceAgent()
    environment = DistillationSequenceEnvironment()
    state, mask = environment.reset()
    steps = []

    # Record current-policy log probabilities before each real environment step.
    for action_id in (1, 4, 8):
        logits = agent.policy(torch.tensor(state, dtype=torch.float32))
        log_probability = masked_distribution(logits, mask).log_prob(torch.tensor(action_id - 1))
        result = environment.step(action_id)
        steps.append(TrajectoryStep(state, action_id, result.reward, log_probability))
        state, mask = result.state, result.action_mask

    episode = EpisodeTrajectory(tuple(steps), result.terminated)
    returns = reward_to_go(step.reward for step in steps)
    expected = -sum(step.log_probability.item() * value for step, value in zip(steps, returns))
    before = [parameter.detach().clone() for parameter in agent.policy.parameters()]
    loss = agent.update((episode,))

    assert loss == pytest.approx(expected, abs=1e-5)
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.policy.parameters()))


def test_agent_update_averages_complete_episode_losses() -> None:
    """Check that two episode losses are averaged without a learned baseline.

    Inputs:
        Two complete legal paths with fixed differentiable log probabilities.
    Returns:
        None. Reported batch loss equals the arithmetic mean of both paths.
    """

    agent = ReinforceAgent()
    log_probability = torch.tensor(-0.5, requires_grad=True)
    first = EpisodeTrajectory((
        TrajectoryStep(INITIAL_STATE, 1, -1.553400, log_probability),
        TrajectoryStep(encode_state(("BCD",)), 4, -1.357200, log_probability),
        TrajectoryStep(encode_state(("CD",)), 8, -1.016760, log_probability),
    ), True)
    second = EpisodeTrajectory((
        TrajectoryStep(INITIAL_STATE, 2, -1.654600, log_probability),
        TrajectoryStep(encode_state(("AB", "CD")), 10, -0.636970, log_probability),
        TrajectoryStep(encode_state(("CD",)), 8, -1.016760, log_probability),
    ), True)

    # A fixed negative log probability makes the expected batch mean explicit.
    first_loss = 0.5 * sum(reward_to_go(step.reward for step in first.steps))
    second_loss = 0.5 * sum(reward_to_go(step.reward for step in second.steps))
    assert agent.update((first, second)) == pytest.approx((first_loss + second_loss) / 2)


def test_agent_update_requires_at_least_one_complete_episode() -> None:
    """Check that an empty update cannot silently skip policy learning.

    Inputs:
        Empty sequence of trajectories.
    Returns:
        None. The update raises ValueError.
    """

    with pytest.raises(ValueError, match="at least one episode"):
        ReinforceAgent().update(())


def test_agent_rejects_masks_inconsistent_with_the_chemical_state() -> None:
    """Check that agent-facing masks cannot enable an impossible tower.

    Inputs:
        Initial ABCD state paired with a mask that allows only tower ten.
    Returns:
        None. Sampling, greedy selection, and reporting reject the mismatch.
    """

    agent = ReinforceAgent()
    wrong_mask = (False,) * 9 + (True,)

    for method in (agent.sample_action, agent.greedy_action, agent.action_probabilities):
        with pytest.raises(ValueError, match="does not match"):
            method(INITIAL_STATE, wrong_mask)


def test_trajectory_validation_rejects_disconnected_states() -> None:
    """Check that individually legal towers still form one physical trajectory.

    Inputs:
        First split to BCD followed by a tower requiring absent ABC.
    Returns:
        None. The disconnected sequence raises ValueError.
    """

    log_probability = torch.tensor(-0.5, requires_grad=True)
    episode = EpisodeTrajectory((
        TrajectoryStep(INITIAL_STATE, 1, -1.553400, log_probability),
        TrajectoryStep(encode_state(("ABC",)), 6, -1.426760, log_probability),
        TrajectoryStep(encode_state(("BC",)), 9, -0.915020, log_probability),
    ), True)

    with pytest.raises(ValueError, match="deterministic sharp splits"):
        _validate_trajectory(episode)


def test_trajectory_validation_rejects_nonfinite_log_probability() -> None:
    """Check that NaN likelihoods cannot contaminate policy parameters.

    Inputs:
        A complete legal trajectory with a nonfinite first log probability.
    Returns:
        None. Validation raises ValueError before optimizer mutation.
    """

    finite = torch.tensor(-0.5, requires_grad=True)
    not_finite = torch.tensor(float("nan"), requires_grad=True)
    episode = EpisodeTrajectory((
        TrajectoryStep(INITIAL_STATE, 1, -1.553400, not_finite),
        TrajectoryStep(encode_state(("BCD",)), 4, -1.357200, finite),
        TrajectoryStep(encode_state(("CD",)), 8, -1.016760, finite),
    ), True)

    with pytest.raises(ValueError, match="nonfinite log probability"):
        _validate_trajectory(episode)


def test_sampled_episode_can_update_policy_end_to_end() -> None:
    """Check that real masked samples remain connected through one Adam step.

    Inputs:
        One sampled three-action episode from the deterministic environment.
    Returns:
        None. The trajectory terminates and yields a finite policy loss.
    """

    agent = ReinforceAgent(ReinforceConfig(seed=7))
    environment = DistillationSequenceEnvironment()
    state, mask = environment.reset()
    steps = []

    # Collect one complete on-policy episode before changing network weights.
    for _ in range(3):
        action_id, log_probability = agent.sample_action(state, mask)
        result = environment.step(action_id)
        steps.append(TrajectoryStep(state, action_id, result.reward, log_probability))
        state, mask = result.state, result.action_mask

    assert result.terminated
    assert torch.isfinite(torch.tensor(agent.update((EpisodeTrajectory(tuple(steps), True),))))
