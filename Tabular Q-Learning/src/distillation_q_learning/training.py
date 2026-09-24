"""
Last modified time: 2026-09-24
Last modified content: Reuse shared evaluation while preserving policy semantics
Last modified by: OpenAI Codex
File design: Episode orchestration around the environment and Q-learning agent
File purpose: Train and evaluate the four-component distillation policy
File creator: OpenAI Codex
"""

from __future__ import annotations

from distillation_sequencing_env.data import EPISODE_STEPS
from distillation_sequencing_env.evaluation import evaluate_greedy
from distillation_sequencing_env.environment import DistillationSequenceEnvironment
from distillation_q_learning.models import EvaluationResult
from distillation_q_learning.q_learning import (
    QLearningConfig,
    TabularQLearningAgent,
    epsilon_for_episode,
)


DEFAULT_EPISODES = 10_000



def _run_training_episode(
    agent: TabularQLearningAgent,
    environment: DistillationSequenceEnvironment,
    epsilon: float,
) -> None:
    """Run one training episode and require exactly three transitions.

    Inputs:
        agent: Q-learning agent to select actions and update.
        environment: Distillation environment to reset and advance.
        epsilon: Exploration probability for the complete episode.
    Returns:
        None. Raises RuntimeError when termination does not occur at step three.
    """

    state, _ = environment.reset()

    # Every valid four-component sharp-split sequence must use exactly three columns.
    for step_number in range(1, EPISODE_STEPS + 1):
        action_id = agent.select_action(state, epsilon)
        result = environment.step(action_id)
        if result.terminated and step_number != EPISODE_STEPS:
            raise RuntimeError("An episode terminated before exactly three steps.")
        agent.update(state, action_id, result.reward, result.state, result.terminated)
        state = result.state
        if result.terminated:
            return

    raise RuntimeError("An episode did not terminate after exactly three steps.")


def train(
    config: QLearningConfig | None = None,
    episodes: int = DEFAULT_EPISODES,
) -> TabularQLearningAgent:
    """Train a tabular Q-learning agent on independent episodes.

    Inputs:
        config: Optional validated Q-learning configuration.
        episodes: Positive number of training episodes.
    Returns:
        Trained tabular Q-learning agent.
    """

    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an integer.")
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    active_config = config or QLearningConfig()
    agent = TabularQLearningAgent(active_config)
    environment = DistillationSequenceEnvironment()

    # Epsilon is fixed within an episode and decays only between episodes.
    for episode in range(episodes):
        epsilon = epsilon_for_episode(active_config, episode)
        _run_training_episode(agent, environment, epsilon)

    return agent


def evaluate(agent: TabularQLearningAgent) -> EvaluationResult:
    """Evaluate the learned deterministic pure strategy without updating Q values.

    Inputs:
        agent: Trained tabular Q-learning agent.
    Returns:
        Immutable evaluation result for one three-column greedy trajectory.
    """


    return evaluate_greedy(lambda state, mask: agent.greedy_action(state), DistillationSequenceEnvironment())
