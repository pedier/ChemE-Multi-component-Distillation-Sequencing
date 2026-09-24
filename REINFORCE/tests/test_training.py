"""
Last modified time: 2026-09-23
Last modified content: Test on-policy training, pure evaluation, and policy reports
Last modified by: OpenAI Codex
File design: Stage-three training and evaluation unit tests
File purpose: Verify complete episodes, batch updates, reproducibility, and economics
File creator: OpenAI Codex
"""

from __future__ import annotations

import pytest
import torch

from distillation_reinforce import (
    DEFAULT_EPISODES,
    EPISODE_STEPS,
    INITIAL_STATE,
    TERMINAL_STATE,
    ReinforceAgent,
    ReinforceConfig,
    action_mask,
    available_actions,
    encode_state,
    evaluate,
    policy_snapshot,
    reachable_states,
    train,
    training_summary,
)
from distillation_reinforce.training import _run_episode


def test_training_constants_and_reachable_state_enumeration() -> None:
    """Check the fixed horizon and all seven reachable policy states.

    Inputs:
        Deterministic textbook transition graph.
    Returns:
        None. State count, uniqueness, and legal pair count match the model.
    """

    states = reachable_states()
    assert DEFAULT_EPISODES == 20_000
    assert EPISODE_STEPS == 3
    assert states[0] == INITIAL_STATE
    assert len(states) == len(set(states)) == 7
    assert TERMINAL_STATE not in states
    assert encode_state(("AB", "CD")) in states
    assert sum(len(available_actions(state)) for state in states) == 12


def test_run_episode_collects_three_connected_on_policy_steps() -> None:
    """Check that a sampled episode carries complete rewards and log likelihoods.

    Inputs:
        Fresh agent with a fixed seed.
    Returns:
        None. Three legal steps terminate with reward equal to negative cost.
    """

    episode, total_reward = _run_episode(ReinforceAgent(ReinforceConfig(seed=7)))
    assert episode.terminated
    assert len(episode.steps) == 3
    assert episode.steps[0].state == INITIAL_STATE
    assert total_reward == pytest.approx(sum(step.reward for step in episode.steps))

    for step in episode.steps:
        assert step.action_id in available_actions(step.state)
        assert step.log_probability.requires_grad


@pytest.mark.parametrize("episodes", [0, -1, 1.5, True])
def test_train_rejects_nonpositive_or_noninteger_episode_counts(episodes: object) -> None:
    """Check that training cannot silently skip or fractionalize episodes.

    Inputs:
        episodes: Invalid requested training count.
    Returns:
        None. The training interface raises ValueError.
    """

    with pytest.raises(ValueError, match="positive integer"):
        train(episodes=episodes)  # type: ignore[arg-type]


def test_train_updates_only_after_complete_batches_and_flushes_remainder() -> None:
    """Check online batch boundaries and per-episode diagnostics.

    Inputs:
        Five episodes with batches of two complete trajectories.
    Returns:
        None. Two full updates and one final partial update are recorded.
    """

    config = ReinforceConfig(batch_episodes=2, seed=11)
    agent = train(config=config, episodes=5)

    assert agent.config == config
    assert len(agent.training_returns) == 5
    assert len(agent.training_losses) == 3
    assert all(-4.574 <= value <= -3.308 for value in agent.training_returns)
    assert all(torch.isfinite(torch.tensor(value)) for value in agent.training_losses)


def test_train_repeats_returns_and_weights_with_the_same_seed() -> None:
    """Check reproducibility of a short CPU training experiment.

    Inputs:
        Two independent eight-episode runs with identical settings.
    Returns:
        None. Return histories, weights, and pure decisions match exactly.
    """

    config = ReinforceConfig(batch_episodes=4, seed=21)
    first = train(config, episodes=8)
    second = train(config, episodes=8)

    assert first.training_returns == second.training_returns
    assert first.training_losses == second.training_losses
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.policy.parameters(), second.policy.parameters())
    )
    assert evaluate(first) == evaluate(second)


def test_evaluate_uses_pure_strategy_without_changing_randomness_or_weights() -> None:
    """Check deterministic ties, complete economics, and read-only evaluation.

    Inputs:
        Agent whose ten logits are equal at every state.
    Returns:
        None. The smallest legal towers yield the known second-best flowsheet.
    """

    agent = ReinforceAgent()
    with torch.no_grad():
        for parameter in agent.policy.parameters():
            parameter.zero_()

    random_state = torch.get_rng_state().clone()
    weights = [parameter.detach().clone() for parameter in agent.policy.parameters()]
    result = evaluate(agent)

    assert result.final_state == TERMINAL_STATE
    assert result.action_sequence == (1, 4, 8)
    assert result.normalized_column_set == (1, 4, 8)
    assert result.total_cost_musd_per_year == pytest.approx(3.927360)
    assert result.total_reward == pytest.approx(-3.927360)
    assert result.selection_vector == (1, 0, 0, 1, 0, 0, 0, 1, 0, 0)
    assert torch.equal(random_state, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(weights, agent.policy.parameters()))


def test_policy_snapshot_reports_seven_masked_pure_choices() -> None:
    """Check complete policy reporting without sampling or changing weights.

    Inputs:
        Fresh policy over all reachable nonterminal states.
    Returns:
        None. Seven records contain only feasible probability mass.
    """

    agent = ReinforceAgent()
    random_state = torch.get_rng_state().clone()
    records = policy_snapshot(agent)

    assert len(records) == 7
    assert records[0]["state"] == INITIAL_STATE
    for record in records:
        state = record["state"]
        mask = action_mask(state)
        probabilities = record["probabilities"]
        assert sum(probabilities) == pytest.approx(1.0)
        assert record["greedy_action"] in record["legal_actions"]
        assert all(probabilities[index] == 0.0 for index in range(10) if not mask[index])

    assert torch.equal(random_state, torch.get_rng_state())


def test_training_summary_groups_returns_and_counts_updates() -> None:
    """Check fixed 1,000-episode learning-curve windows.

    Inputs:
        Synthetic 1,002-episode return history and two update losses.
    Returns:
        None. Episode counts, endpoint means, and window means are exact.
    """

    agent = ReinforceAgent()
    agent.training_returns = [-1.0] * 1_000 + [-2.0, -2.0]
    agent.training_losses = [0.1, 0.2]
    summary = training_summary(agent)

    assert summary["episodes"] == 1_002
    assert summary["updates"] == 2
    assert summary["mean_reward_first_100"] == pytest.approx(-1.0)
    assert summary["mean_reward_last_100"] == pytest.approx(-1.02)
    assert summary["mean_reward_by_1000"] == pytest.approx([-1.0, -2.0])


def test_training_summary_rejects_untrained_agent() -> None:
    """Check that a missing learning curve is reported explicitly.

    Inputs:
        Fresh agent with no online training returns.
    Returns:
        None. Summary generation raises ValueError.
    """

    with pytest.raises(ValueError, match="history is empty"):
        training_summary(ReinforceAgent())
