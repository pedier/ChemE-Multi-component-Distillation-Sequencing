"""
Last modified time: 2026-09-24
Last modified content: Reuse shared evaluation while preserving policy semantics
Last modified by: OpenAI Codex
File design: Complete episode collection and deterministic result reporting
File purpose: Connect masked PPO to the four-component chemical environment
File creator: OpenAI Codex
"""

from __future__ import annotations

from distillation_sequencing_env.evaluation import (
    evaluate_greedy, policy_records, reachable_states as _reachable_states,
)

from distillation_sequencing_env.data import EPISODE_STEPS

from dataclasses import asdict

from distillation_sequencing_env.data import TERMINAL_STATE
from distillation_sequencing_env.environment import (
    DistillationSequenceEnvironment,
)
from distillation_ppo.models import EvaluationResult, State
from distillation_ppo.ppo import EpisodeTrajectory, PPOAgent, PPOConfig, RolloutStep


DEFAULT_EPISODES = 20_000

SUMMARY_WINDOW = 1_000


def _run_episode(agent: PPOAgent) -> tuple[EpisodeTrajectory, float]:
    """Collect one complete three-tower episode under unchanged PPO weights.

    Inputs:
        agent: Current actor-critic used to sample all three legal towers.
    Returns:
        Fresh rollout trajectory and its negative annualized total cost.
    """

    environment = DistillationSequenceEnvironment()
    state, mask = environment.reset()
    steps: list[RolloutStep] = []

    # Keep behavior-policy log probabilities and values fixed for all three decisions.
    for index in range(EPISODE_STEPS):
        sample = agent.sample_action(state, mask)
        result = environment.step(sample.action_id)
        steps.append(RolloutStep(
            state, mask, sample, result.reward, result.state, result.terminated,
        ))
        state, mask = result.state, result.action_mask
        if result.terminated and index != EPISODE_STEPS - 1:
            raise RuntimeError("A separation episode ended before three towers.")

    if not result.terminated or state != TERMINAL_STATE:
        raise RuntimeError("A separation episode did not finish after three towers.")

    return EpisodeTrajectory(tuple(steps)), sum(step.reward for step in steps)


def train(config: PPOConfig | None = None, episodes: int = DEFAULT_EPISODES) -> PPOAgent:
    """Train PPO from new complete on-policy episode batches.

    Inputs:
        config: Optional validated PPO settings.
        episodes: Positive number of complete three-step episodes.
    Returns:
        Trained agent with sampled returns and update diagnostics.
    """

    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer.")

    agent = PPOAgent(config)
    batch: list[EpisodeTrajectory] = []

    # Update only after collecting a batch under one unchanged policy version.
    for _ in range(episodes):
        trajectory, total_reward = _run_episode(agent)
        batch.append(trajectory)
        agent.training_returns.append(total_reward)

        if len(batch) == agent.config.batch_episodes:
            agent.update_history.append(agent.update(batch))
            batch.clear()

    if batch:
        agent.update_history.append(agent.update(batch))

    return agent


def evaluate(agent: PPOAgent) -> EvaluationResult:
    """Evaluate one deterministic pure flowsheet without policy updates.

    Inputs:
        agent: Trained or newly initialized PPO actor-critic.
    Returns:
        Ordered towers, canonical tower set, total cost, and total reward.
    """


    return evaluate_greedy(agent.greedy_action, DistillationSequenceEnvironment())


def reachable_states() -> tuple[State, ...]:
    """Enumerate every nonterminal state reachable from the textbook feed.

    Inputs:
        None. Uses the fixed deterministic sharp-split environment.
    Returns:
        Seven unique nonterminal states in breadth-first discovery order.
    """


    return _reachable_states()


def policy_snapshot(agent: PPOAgent) -> tuple[dict[str, object], ...]:
    """Report legal probabilities and pure choices for all reachable states.

    Inputs:
        agent: PPO policy to inspect without changing weights or random state.
    Returns:
        JSON-ready record for each reachable nonterminal chemical state.
    """


    return policy_records(agent.inspect_policy)


def training_summary(agent: PPOAgent) -> dict[str, object]:
    """Summarize sampled rewards and optimization diagnostics.

    Inputs:
        agent: PPO agent returned by a completed training run.
    Returns:
        JSON-ready learning-curve means and final PPO update statistics.
    """

    returns = agent.training_returns
    if not returns or not agent.update_history:
        raise ValueError("Training history is empty.")

    first = returns[:100]
    last = returns[-100:]
    windows = [
        sum(returns[index:index + SUMMARY_WINDOW]) / len(returns[index:index + SUMMARY_WINDOW])
        for index in range(0, len(returns), SUMMARY_WINDOW)
    ]
    return {
        "episodes": len(returns),
        "updates": len(agent.update_history),
        "mean_reward_first_100": sum(first) / len(first),
        "mean_reward_last_100": sum(last) / len(last),
        "mean_reward_by_1000": windows,
        "last_update": asdict(agent.update_history[-1]),
    }
