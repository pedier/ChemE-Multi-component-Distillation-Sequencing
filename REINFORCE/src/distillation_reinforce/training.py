"""
Last modified time: 2026-09-24
Last modified content: Reuse shared evaluation while preserving policy semantics
Last modified by: OpenAI Codex
File design: Complete episode collection, training, and deterministic reporting
File purpose: Connect the masked policy to the four-component chemical environment
File creator: OpenAI Codex
"""

from __future__ import annotations

from distillation_sequencing_env.evaluation import (
    evaluate_greedy, policy_records, reachable_states as _reachable_states,
)

from distillation_sequencing_env.data import EPISODE_STEPS


from distillation_sequencing_env.data import TERMINAL_STATE
from distillation_sequencing_env.environment import (
    DistillationSequenceEnvironment,
)
from distillation_reinforce.models import EvaluationResult, State
from distillation_reinforce.reinforce import (
    EpisodeTrajectory,
    ReinforceAgent,
    ReinforceConfig,
    TrajectoryStep,
)


DEFAULT_EPISODES = 20_000

SUMMARY_WINDOW = 1_000


def _run_episode(agent: ReinforceAgent) -> tuple[EpisodeTrajectory, float]:
    """Collect one complete on-policy three-tower separation sequence.

    Inputs:
        agent: Current masked REINFORCE policy used for all three samples.
    Returns:
        Complete trajectory and its negative annualized total cost.
    """

    environment = DistillationSequenceEnvironment()
    state, mask = environment.reset()
    steps: list[TrajectoryStep] = []

    # Keep all three action log probabilities attached to the current policy.
    for index in range(EPISODE_STEPS):
        action_id, log_probability = agent.sample_action(state, mask)
        result = environment.step(action_id)
        steps.append(TrajectoryStep(state, action_id, result.reward, log_probability))
        state, mask = result.state, result.action_mask

        if result.terminated and index != EPISODE_STEPS - 1:
            raise RuntimeError("A separation episode ended before three towers.")

    if not result.terminated or state != TERMINAL_STATE:
        raise RuntimeError("A separation episode did not finish after three towers.")

    episode = EpisodeTrajectory(tuple(steps), True)
    return episode, sum(step.reward for step in steps)


def train(
    config: ReinforceConfig | None = None, episodes: int = DEFAULT_EPISODES
) -> ReinforceAgent:
    """Train a policy from fresh on-policy complete episodes.

    Inputs:
        config: Optional optimizer, batch-size, and seed settings.
        episodes: Positive number of training episodes.
    Returns:
        Trained agent with episode returns and update losses for reporting.
    """

    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer.")

    agent = ReinforceAgent(config)
    batch: list[EpisodeTrajectory] = []

    # Update only after gathering a batch under unchanged policy weights.
    for _ in range(episodes):
        trajectory, total_reward = _run_episode(agent)
        batch.append(trajectory)
        agent.training_returns.append(total_reward)

        if len(batch) == agent.config.batch_episodes:
            agent.training_losses.append(agent.update(batch))
            batch.clear()

    if batch:
        agent.training_losses.append(agent.update(batch))

    return agent


def evaluate(agent: ReinforceAgent) -> EvaluationResult:
    """Run one deterministic pure-policy flowsheet without sampling or updates.

    Inputs:
        agent: Trained or initialized REINFORCE policy.
    Returns:
        Complete tower sequence, normalized tower set, cost, and reward.
    """


    return evaluate_greedy(agent.greedy_action, DistillationSequenceEnvironment())


def reachable_states() -> tuple[State, ...]:
    """Enumerate every nonterminal state reachable from the textbook feed.

    Inputs:
        None. Uses fixed deterministic sharp-split transitions.
    Returns:
        Seven nonterminal states in breadth-first discovery order.
    """


    return _reachable_states()


def policy_snapshot(agent: ReinforceAgent) -> tuple[dict[str, object], ...]:
    """Report masked probabilities and pure choices at reachable states.

    Inputs:
        agent: Policy to inspect without changing weights or random state.
    Returns:
        One JSON-ready record for each reachable nonterminal state.
    """


    return policy_records(agent.inspect_policy)


def training_summary(agent: ReinforceAgent) -> dict[str, object]:
    """Summarize sampled return history without changing the policy.

    Inputs:
        agent: Agent with episode returns and update losses from train.
    Returns:
        JSON-ready episode, update, and mean-return learning-curve data.
    """

    returns = agent.training_returns
    if not returns:
        raise ValueError("Training history is empty.")

    first = returns[:100]
    last = returns[-100:]
    windows = [
        sum(returns[index:index + SUMMARY_WINDOW]) / len(returns[index:index + SUMMARY_WINDOW])
        for index in range(0, len(returns), SUMMARY_WINDOW)
    ]
    return {
        "episodes": len(returns),
        "updates": len(agent.training_losses),
        "mean_reward_first_100": sum(first) / len(first),
        "mean_reward_last_100": sum(last) / len(last),
        "mean_reward_by_1000": windows,
    }
