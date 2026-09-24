"""
Last modified time: 2026-09-24
Last modified content: Verify masked sampling, rollout validation, and fresh PPO updates
Last modified by: OpenAI Codex
File design: Actor-critic agent behavior tests
File purpose: Check on-policy freshness, physical trajectories, and optimization
File creator: OpenAI Codex
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from distillation_ppo import DistillationSequenceEnvironment, action_mask
from distillation_ppo.data import INITIAL_STATE, TERMINAL_STATE
from distillation_ppo.ppo import (
    ActionSample,
    EpisodeTrajectory,
    PPOAgent,
    PPOConfig,
    RolloutStep,
    validate_episode,
)


def make_episode(agent: PPOAgent) -> EpisodeTrajectory:
    """Collect one complete legal trajectory under the current PPO policy.

    Inputs:
        agent: Actor-critic whose sampled actions and values are recorded.
    Returns:
        Connected three-step episode with true chemical costs and terminal flag.
    """

    environment = DistillationSequenceEnvironment()
    state, mask = environment.reset()
    steps = []

    # Sampling and environment transitions remain separate so stored statistics are frozen.
    for _ in range(3):
        sample = agent.sample_action(state, mask)
        result = environment.step(sample.action_id)
        steps.append(RolloutStep(state, mask, sample, result.reward, result.state, result.terminated))
        state, mask = result.state, result.action_mask

    assert state == TERMINAL_STATE
    return EpisodeTrajectory(tuple(steps))


def test_sample_action_freezes_legal_probabilities_and_old_value() -> None:
    """Verify sampled tower IDs and behavior-policy statistics are valid.

    Inputs:
        Initial chemical state, legal mask, and seeded PPO agent.
    Returns:
        None. Sampled actions are legal, finite, and detached from gradients.
    """

    agent = PPOAgent()
    mask = action_mask(INITIAL_STATE)
    probabilities = agent.action_probabilities(INITIAL_STATE, mask)
    samples = [agent.sample_action(INITIAL_STATE, mask) for _ in range(30)]
    assert {sample.action_id for sample in samples} <= {1, 2, 3}
    assert all(sample.policy_version == 0 for sample in samples)
    assert all(torch.isfinite(torch.tensor(sample.old_value)) for sample in samples)
    for sample in samples:
        assert sample.old_log_probability == pytest.approx(
            torch.log(torch.tensor(probabilities[sample.action_id - 1])).item()
        )


def test_greedy_action_breaks_ties_and_probabilities_respect_mask() -> None:
    """Verify pure selection and zero probability for infeasible towers.

    Inputs:
        Initial state and AB-plus-CD state under a zero-logit actor.
    Returns:
        None. Greedy decisions use the lowest legal ID on exact ties.
    """

    agent = PPOAgent()
    for parameter in agent.actor.parameters():
        torch.nn.init.zeros_(parameter)

    initial_mask = action_mask(INITIAL_STATE)
    assert agent.greedy_action(INITIAL_STATE, initial_mask) == 1
    probabilities = agent.action_probabilities(INITIAL_STATE, initial_mask)
    assert probabilities[:3] == pytest.approx((1 / 3,) * 3)
    assert probabilities[3:] == (0.0,) * 7

    branch_state = (0, 0, 0, 1, 0, 1)
    assert agent.greedy_action(branch_state, action_mask(branch_state)) == 8


def test_agent_distribution_rejects_mask_mismatch_and_terminal_state() -> None:
    """Verify the private distribution builder enforces environment masks.

    Inputs:
        Initial and terminal states with incorrect or empty masks.
    Returns:
        None. Both invalid requests raise ValueError.
    """

    agent = PPOAgent()
    with pytest.raises(ValueError):
        agent._distribution(INITIAL_STATE, (False,) * 10)
    with pytest.raises(ValueError):
        agent.sample_action(TERMINAL_STATE, action_mask(TERMINAL_STATE))


def test_validate_episode_accepts_connected_physical_rollout() -> None:
    """Verify one sampled episode preserves actions, costs, and termination.

    Inputs:
        Newly collected on-policy trajectory.
    Returns:
        None. Validation succeeds and total reward equals negative cost.
    """

    agent = PPOAgent()
    episode = make_episode(agent)
    validate_episode(episode, agent.policy_version)
    assert len(episode.steps) == 3
    assert episode.steps[-1].terminated is True
    assert episode.steps[-1].next_state == TERMINAL_STATE


@pytest.mark.parametrize("corruption", [
    "wrong_start", "wrong_mask", "stale", "bad_action", "bad_reward",
    "bad_next_state", "early_terminal", "nonfinite_value", "short_episode",
])
def test_validate_episode_rejects_invalid_or_stale_rollouts(corruption: str) -> None:
    """Verify corrupt chemical or behavior-policy records cannot reach PPO.

    Inputs:
        One valid rollout and a selected record corruption.
    Returns:
        None. Validation raises ValueError for every corruption.
    """

    agent = PPOAgent()
    episode = make_episode(agent)
    steps = list(episode.steps)
    first = steps[0]
    if corruption == "wrong_start":
        steps[0] = replace(first, state=TERMINAL_STATE)
    elif corruption == "wrong_mask":
        steps[0] = replace(first, mask=(False,) * 10)
    elif corruption == "stale":
        steps[0] = replace(first, sample=replace(first.sample, policy_version=1))
    elif corruption == "bad_action":
        steps[0] = replace(first, sample=replace(first.sample, action_id=11))
    elif corruption == "bad_reward":
        steps[0] = replace(first, reward=0.0)
    elif corruption == "bad_next_state":
        steps[0] = replace(first, next_state=TERMINAL_STATE)
    elif corruption == "early_terminal":
        steps[0] = replace(first, terminated=True)
    elif corruption == "nonfinite_value":
        steps[0] = replace(first, sample=replace(first.sample, old_value=float("nan")))
    elif corruption == "short_episode":
        steps.pop()

    with pytest.raises(ValueError):
        validate_episode(EpisodeTrajectory(tuple(steps)), agent.policy_version)


def test_prepare_batch_normalizes_advantages_and_preserves_old_statistics() -> None:
    """Verify sampled old statistics become fixed PPO tensor targets.

    Inputs:
        Two independently collected episodes under one policy version.
    Returns:
        None. The tensor batch has six transitions and normalized advantages.
    """

    agent = PPOAgent()
    episodes = (make_episode(agent), make_episode(agent))
    batch = agent._prepare_batch(episodes)
    assert batch.states.shape == (6, 6)
    assert batch.masks.shape == (6, 10)
    assert batch.action_indices.shape == (6,)
    assert batch.returns.shape == (6,)
    assert batch.advantages.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert batch.old_log_probabilities[0].item() == pytest.approx(
        episodes[0].steps[0].sample.old_log_probability
    )


def test_prepare_batch_rejects_empty_input() -> None:
    """Verify PPO cannot update from an empty rollout batch.

    Inputs:
        Empty episode tuple.
    Returns:
        None. Batch preparation raises ValueError.
    """

    with pytest.raises(ValueError):
        PPOAgent()._prepare_batch(())


def test_loss_terms_are_finite_and_differentiable() -> None:
    """Verify minibatch actor, critic, entropy, and diagnostics are finite.

    Inputs:
        A fresh three-transition batch and its first two indices.
    Returns:
        None. Terms are finite and joint optimization has gradients.
    """

    agent = PPOAgent()
    batch = agent._prepare_batch((make_episode(agent),))
    terms = agent._loss_terms(batch, torch.tensor([0, 1]))
    assert len(terms) == 5
    assert all(bool(torch.isfinite(term)) for term in terms)
    (terms[0] + terms[1]).backward()
    assert any(parameter.grad is not None for parameter in agent.critic.parameters())


def test_update_reuses_fresh_batch_then_rejects_stale_episodes() -> None:
    """Verify multiple PPO minibatches change weights only once per policy version.

    Inputs:
        One complete episode and two optimization epochs of size two.
    Returns:
        None. Four optimizer steps occur, diagnostics are finite, and reuse fails.
    """

    agent = PPOAgent(PPOConfig(update_epochs=2, minibatch_size=2))
    episode = make_episode(agent)
    old_weights = [parameter.detach().clone() for parameter in agent.critic.parameters()]
    old_log_probabilities = tuple(step.sample.old_log_probability for step in episode.steps)
    result = agent.update((episode,))

    assert result.transitions == 3
    assert result.minibatches == 4
    assert result.approximate_kl >= 0.0
    assert 0.0 <= result.clip_fraction <= 1.0
    assert agent.policy_version == 1
    assert any(not torch.equal(old, new) for old, new in zip(old_weights, agent.critic.parameters()))
    assert old_log_probabilities == tuple(step.sample.old_log_probability for step in episode.steps)
    with pytest.raises(ValueError, match="stale"):
        agent.update((episode,))


def test_update_rejects_empty_input_without_changing_policy() -> None:
    """Verify an empty update leaves network version unchanged.

    Inputs:
        Empty episode tuple and a new PPO agent.
    Returns:
        None. Update raises ValueError before optimizer work.
    """

    agent = PPOAgent()
    with pytest.raises(ValueError):
        agent.update(())
    assert agent.policy_version == 0


def test_action_sample_and_rollout_models_are_immutable() -> None:
    """Verify behavior-policy metadata cannot be changed after sampling.

    Inputs:
        An ActionSample and an EpisodeTrajectory.
    Returns:
        None. Assignment to frozen records fails.
    """

    from dataclasses import FrozenInstanceError

    sample = ActionSample(1, 0.0, 0.0, 0)
    with pytest.raises(FrozenInstanceError):
        sample.action_id = 2  # type: ignore[misc]
    episode = make_episode(PPOAgent())
    with pytest.raises(FrozenInstanceError):
        episode.steps = ()  # type: ignore[misc]
