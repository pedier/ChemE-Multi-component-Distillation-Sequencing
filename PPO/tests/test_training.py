"""
Last modified time: 2026-09-24
Last modified content: Verify fresh PPO training, deterministic evaluation, and policy summaries
Last modified by: OpenAI Codex
File design: Training and engineering-result integration tests
File purpose: Check three-step episodes, batch updates, reproducibility, and pure flowsheets
File creator: OpenAI Codex
"""

from __future__ import annotations

import pytest
import torch

from distillation_ppo import TERMINAL_STATE, action_mask
from distillation_ppo.environment import available_actions
from distillation_ppo.ppo import PPOAgent, PPOConfig, validate_episode
from distillation_ppo.training import (
    DEFAULT_EPISODES,
    _run_episode,
    evaluate,
    policy_snapshot,
    reachable_states,
    train,
    training_summary,
)


def test_run_episode_collects_three_connected_on_policy_splits() -> None:
    """Verify all rollout steps use one policy version and true tower rewards.

    Inputs:
        Seeded PPO agent and a fresh chemical environment per episode.
    Returns:
        None. The episode validates and ends after three legal actions.
    """

    agent = PPOAgent()
    trajectory, total_reward = _run_episode(agent)
    validate_episode(trajectory, agent.policy_version)
    assert len(trajectory.steps) == 3
    assert trajectory.steps[-1].next_state == TERMINAL_STATE
    assert total_reward == pytest.approx(sum(step.reward for step in trajectory.steps))
    assert all(step.sample.policy_version == 0 for step in trajectory.steps)


@pytest.mark.parametrize("episodes", [0, -1, True, 1.5])
def test_train_rejects_invalid_episode_counts(episodes: object) -> None:
    """Verify training requires a positive integer number of episodes.

    Inputs:
        Invalid episode count.
    Returns:
        None. Training raises ValueError before constructing an agent.
    """

    with pytest.raises(ValueError):
        train(episodes=episodes)  # type: ignore[arg-type]


def test_train_uses_fresh_full_and_partial_batches() -> None:
    """Verify on-policy batches expire after each update, including the last partial batch.

    Inputs:
        Nine episodes with four episodes per intended batch.
    Returns:
        None. Three updates consume 27 transitions and preserve rewards.
    """

    config = PPOConfig(batch_episodes=4, update_epochs=1, minibatch_size=12)
    agent = train(config, episodes=9)
    assert len(agent.training_returns) == 9
    assert len(agent.update_history) == 3
    assert agent.policy_version == 3
    assert [item.transitions for item in agent.update_history] == [12, 12, 3]
    assert all(-4.6 < reward < -3.3 for reward in agent.training_returns)


def test_train_is_reproducible_for_the_same_seed() -> None:
    """Verify identical settings produce identical sampled and pure results.

    Inputs:
        Two independent short PPO training runs with seed 7.
    Returns:
        None. Returns, update diagnostics, and pure flowsheets agree.
    """

    config = PPOConfig(seed=7, batch_episodes=4, update_epochs=1, minibatch_size=12)
    first = train(config, episodes=8)
    second = train(config, episodes=8)
    assert first.training_returns == second.training_returns
    assert first.update_history == second.update_history
    assert evaluate(first) == evaluate(second)


def test_evaluate_is_pure_read_only_and_reward_matches_cost() -> None:
    """Verify evaluation neither samples nor alters actor-critic parameters.

    Inputs:
        Initialized PPO agent and its current random state and weights.
    Returns:
        None. A legal three-tower pure flowsheet has exact negative cost.
    """

    agent = PPOAgent()
    random_state = torch.get_rng_state().clone()
    weights = [value.clone() for value in agent.actor.state_dict().values()]
    result = evaluate(agent)
    assert result.final_state == TERMINAL_STATE
    assert len(result.action_sequence) == 3
    assert result.normalized_column_set == tuple(sorted(result.action_sequence))
    assert result.total_reward == pytest.approx(-result.total_cost_musd_per_year)
    assert agent.policy_version == 0
    assert torch.equal(torch.get_rng_state(), random_state)
    assert all(torch.equal(old, new) for old, new in zip(weights, agent.actor.state_dict().values()))


def test_reachable_states_enumerates_every_nonterminal_decision_state() -> None:
    """Verify state enumeration includes each feasible decision point once.

    Inputs:
        Fixed textbook sharp-split state graph.
    Returns:
        None. Seven nonterminal states and their legal actions are present.
    """

    states = reachable_states()
    assert len(states) == len(set(states)) == 7
    assert states[0] == (1, 0, 0, 0, 0, 0)
    assert TERMINAL_STATE not in states
    assert {available_actions(state) for state in states} == {
        (1, 2, 3), (4, 5), (6, 7), (8, 10), (8,), (9,), (10,),
    }


def test_policy_snapshot_reports_only_legal_probability_mass() -> None:
    """Verify seven policy rows expose masks, probabilities, and greedy tower IDs.

    Inputs:
        Untrained masked PPO agent.
    Returns:
        None. Each row is normalized over legal towers only.
    """

    agent = PPOAgent()
    records = policy_snapshot(agent)
    assert len(records) == 7

    # Check the reported action probabilities against the environment mask.
    for record in records:
        state = record["state"]
        mask = action_mask(state)
        probabilities = record["probabilities"]
        assert sum(probabilities) == pytest.approx(1.0)
        assert all(probability == 0.0 for probability, legal in zip(probabilities, mask) if not legal)
        assert record["greedy_action"] in record["legal_actions"]


def test_training_summary_rejects_untrained_agent() -> None:
    """Verify training diagnostics require sampled episodes and updates.

    Inputs:
        Newly initialized PPO agent.
    Returns:
        None. Summary raises ValueError.
    """

    with pytest.raises(ValueError):
        training_summary(PPOAgent())


def test_training_summary_reports_learning_curve_and_last_update() -> None:
    """Verify rewards and PPO diagnostics stay separate from chemical cost.

    Inputs:
        Seven-episode seeded run with a partial final batch.
    Returns:
        None. Summary counts updates and reports finite means and diagnostics.
    """

    config = PPOConfig(batch_episodes=4, update_epochs=1, minibatch_size=12)
    agent = train(config, episodes=7)
    summary = training_summary(agent)
    assert summary["episodes"] == 7
    assert summary["updates"] == 2
    assert summary["mean_reward_first_100"] == pytest.approx(sum(agent.training_returns) / 7)
    assert summary["mean_reward_last_100"] == summary["mean_reward_first_100"]
    assert summary["mean_reward_by_1000"] == pytest.approx([sum(agent.training_returns) / 7])
    assert summary["last_update"]["transitions"] == 9
    assert DEFAULT_EPISODES == 20_000
