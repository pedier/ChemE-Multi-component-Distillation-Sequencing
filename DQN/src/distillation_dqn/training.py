"""
Last modified time: 2026-09-24
Last modified content: Reuse shared evaluation while preserving policy semantics
Last modified by: OpenAI Codex
File design: Episode orchestration around the fixed chemical environment
File purpose: Train with observed transitions and audit greedy flowsheet economics
File creator: OpenAI Codex
"""

from __future__ import annotations

from distillation_sequencing_env.data import EPISODE_STEPS
from distillation_sequencing_env.evaluation import evaluate_greedy
from distillation_dqn.agent import DQNConfig, DuelingDoubleDQNAgent
from distillation_sequencing_env.environment import DistillationSequenceEnvironment
from distillation_dqn.models import EvaluationResult
from distillation_dqn.replay import Transition


DEFAULT_EPISODES = 10_000

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995


def epsilon_for_episode(episode: int) -> float:
    """Calculate the episode-indexed bounded exploration probability.

    Inputs:
        episode: Nonnegative zero-based episode index.
    Returns:
        Probability that decays from one toward a floor of 0.05.
    """

    if isinstance(episode, bool) or not isinstance(episode, int):
        raise TypeError("episode must be an integer.")
    if episode < 0:
        raise ValueError("episode must be nonnegative.")

    return max(EPSILON_MIN, EPSILON_START * EPSILON_DECAY**episode)


def _run_training_episode(
    agent: DuelingDoubleDQNAgent,
    environment: DistillationSequenceEnvironment,
    epsilon: float,
) -> None:
    """Collect one genuine three-split episode and update the learner.

    Inputs:
        agent: DQN agent selecting towers and observing transitions.
        environment: Deterministic chemical environment to reset and step.
        epsilon: Fixed exploration probability for this episode.
    Returns:
        None. Raises RuntimeError for an invalid episode length.
    """

    state, mask = environment.reset()

    # Every complete four-component sharp-split sequence uses exactly three towers.
    for step_number in range(1, EPISODE_STEPS + 1):
        action_id = agent.select_action(state, mask, epsilon)
        result = environment.step(action_id)
        if result.terminated and step_number != EPISODE_STEPS:
            raise RuntimeError("An episode terminated before exactly three steps.")
        if step_number == EPISODE_STEPS and not result.terminated:
            raise RuntimeError("An episode did not terminate after exactly three steps.")

        transition = Transition(
            state, action_id, result.reward, result.state, result.action_mask, result.terminated
        )
        agent.observe(transition)
        state, mask = result.state, result.action_mask
        if result.terminated:
            return


def train(
    config: DQNConfig | None = None,
    episodes: int = DEFAULT_EPISODES,
) -> DuelingDoubleDQNAgent:
    """Train the masked DQN solely from online environment interactions.

    Inputs:
        config: Optional validated learning configuration.
        episodes: Positive number of independent three-step episodes.
    Returns:
        Trained Dueling Double DQN agent.
    """

    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an integer.")
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    agent = DuelingDoubleDQNAgent(config)
    environment = DistillationSequenceEnvironment()

    # Epsilon stays fixed within one flowsheet and changes only between episodes.
    for episode in range(episodes):
        _run_training_episode(agent, environment, epsilon_for_episode(episode))

    return agent


def evaluate(agent: DuelingDoubleDQNAgent) -> EvaluationResult:
    """Run one pure greedy flowsheet without exploration or learning.

    Inputs:
        agent: DQN agent whose online network supplies legal Q values.
    Returns:
        Immutable three-tower economic and trajectory result.
    """


    return evaluate_greedy(lambda state, mask: agent.greedy_action(state), DistillationSequenceEnvironment())


