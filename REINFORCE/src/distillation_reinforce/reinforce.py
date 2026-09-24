"""
Last modified time: 2026-09-24
Last modified content: Reuse shared evaluation while preserving policy semantics
Last modified by: OpenAI Codex
File design: Episodic policy-gradient agent and immutable trajectory records
File purpose: Sample legal towers and optimize undiscounted reward-to-go
File creator: OpenAI Codex
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.distributions import Categorical

from distillation_sequencing_env.data import EPISODE_STEPS, INITIAL_STATE, TERMINAL_STATE
from distillation_sequencing_env.environment import action_mask, apply_action, available_actions, validate_reward
from distillation_reinforce.models import ActionMask, State
from distillation_reinforce.policy import PolicyNetwork, masked_distribution


@dataclass(frozen=True)
class ReinforceConfig:
    """Hold the fixed policy-gradient training settings.

    Inputs:
        learning_rate: Adam step size.
        batch_episodes: Number of complete on-policy episodes per update.
        seed: Seed for PyTorch network initialization and action sampling.
    Returns:
        Immutable validated configuration.
    """

    learning_rate: float = 1e-3
    batch_episodes: int = 32
    seed: int = 42

    def __post_init__(self) -> None:
        """Reject invalid optimization settings.

        Inputs:
            None. Reads this instance's fields.
        Returns:
            None. Raises ValueError for invalid settings.
        """

        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive and finite.")

        if (
            isinstance(self.batch_episodes, bool)
            or not isinstance(self.batch_episodes, int)
            or self.batch_episodes <= 0
        ):
            raise ValueError("batch_episodes must be a positive integer.")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")


@dataclass(frozen=True)
class TrajectoryStep:
    """Record one sampled action and its differentiable log probability.

    Inputs:
        state: Six-bit state before the action.
        action_id: Sampled textbook tower ID.
        reward: Negative incremental annual cost in M$/yr.
        log_probability: Scalar log probability under the current policy.
    Returns:
        Immutable step record for one on-policy trajectory.
    """

    state: State
    action_id: int
    reward: float
    log_probability: Tensor


@dataclass(frozen=True)
class EpisodeTrajectory:
    """Store a complete three-action distillation trajectory.

    Inputs:
        steps: Ordered sequence of three sampled trajectory steps.
        terminated: Whether the environment reached four pure products.
    Returns:
        Immutable episode record for a REINFORCE update.
    """

    steps: tuple[TrajectoryStep, ...]
    terminated: bool


def reward_to_go(rewards: Iterable[float]) -> tuple[float, ...]:
    """Compute undiscounted Monte Carlo return from each action onward.

    Inputs:
        rewards: Ordered finite incremental rewards from one episode.
    Returns:
        Tuple of suffix sums with the same length as the rewards.
    """

    values = tuple(rewards)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Rewards must be a nonempty sequence of finite values.")

    # Sum from the terminal action backward without discounting later towers.
    running_total = 0.0
    returns = []
    for value in reversed(values):
        running_total += value
        returns.append(running_total)

    return tuple(reversed(returns))


def _validate_trajectory(episode: EpisodeTrajectory) -> None:
    """Check that an update receives a complete legal sampled episode.

    Inputs:
        episode: Candidate three-step trajectory with log probabilities.
    Returns:
        None. Raises ValueError if it is incomplete or inconsistent.
    """

    if not episode.terminated or len(episode.steps) != EPISODE_STEPS:
        raise ValueError("REINFORCE requires complete three-step episodes.")

    if episode.steps[0].state != INITIAL_STATE:
        raise ValueError("A trajectory must begin with the ABCD feed.")

    # Verify sampled actions and the physical state reached after each split.
    for index, step in enumerate(episode.steps):
        if step.action_id not in available_actions(step.state):
            raise ValueError("A trajectory contains an infeasible tower.")

        validate_reward(step.action_id, step.reward)

        if step.log_probability.ndim != 0 or not step.log_probability.requires_grad:
            raise ValueError("Each log probability must be a scalar with gradients.")

        if not bool(torch.isfinite(step.log_probability)):
            raise ValueError("A trajectory contains a nonfinite log probability.")

        next_state = apply_action(step.state, step.action_id)
        expected = episode.steps[index + 1].state if index < EPISODE_STEPS - 1 else TERMINAL_STATE
        if next_state != expected:
            raise ValueError("Trajectory states must follow deterministic sharp splits.")


class ReinforceAgent:
    """Sample masked actions and apply baseline-free episodic policy gradients.

    Inputs:
        config: Learning rate, batch size, and random seed settings.
    Returns:
        Agent with a six-input policy network and Adam optimizer.
    """

    def __init__(self, config: ReinforceConfig | None = None) -> None:
        """Initialize the CPU policy, optimizer, and reproducible random seed.

        Inputs:
            config: Optional validated REINFORCE settings.
        Returns:
            None. Creates the trainable policy and optimizer.
        """

        self.config = config or ReinforceConfig()
        torch.manual_seed(self.config.seed)
        self.policy = PolicyNetwork().cpu()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.training_returns: list[float] = []
        self.training_losses: list[float] = []

    def _distribution(self, state: State, mask: ActionMask) -> Categorical:
        """Build the current policy distribution for a verified chemical state.

        Inputs:
            state: Current six-bit chemical state.
            mask: Claimed legal-action mask for that state.
        Returns:
            Categorical distribution over exactly the feasible towers.
        """

        if mask != action_mask(state):
            raise ValueError("Action mask does not match the chemical state.")

        state_tensor = torch.tensor(state, dtype=torch.float32)
        return masked_distribution(self.policy(state_tensor), mask)

    def sample_action(self, state: State, mask: ActionMask) -> tuple[int, Tensor]:
        """Sample a feasible tower and retain its differentiable log probability.

        Inputs:
            state: Current six-bit chemical state.
            mask: Legal textbook towers at that state.
        Returns:
            External tower ID and scalar log probability for REINFORCE.
        """

        distribution = self._distribution(state, mask)
        action_index = distribution.sample()
        return int(action_index.item()) + 1, distribution.log_prob(action_index)

    def inspect_policy(self, state: State, mask: ActionMask) -> tuple[tuple[float, ...], int]:
        """Inspect probabilities and a greedy choice with one forward pass.

        Inputs:
            state: Current valid chemical state.
            mask: Legal tower mask matching the state.
        Returns:
            Ten probabilities and the deterministic legal argmax tower ID.
        """

        with torch.no_grad():
            distribution = self._distribution(state, mask)
            probabilities = tuple(float(value) for value in distribution.probs.tolist())
            return probabilities, int(torch.argmax(distribution.logits).item()) + 1

    def greedy_action(self, state: State, mask: ActionMask) -> int:
        """Choose the highest-probability feasible tower without sampling.

        Inputs:
            state: Current six-bit chemical state.
            mask: Legal textbook towers at that state.
        Returns:
            External tower ID, choosing the smaller ID on an exact tie.
        """

        with torch.no_grad():
            distribution = self._distribution(state, mask)
            return int(torch.argmax(distribution.logits).item()) + 1

    def action_probabilities(self, state: State, mask: ActionMask) -> tuple[float, ...]:
        """Report the ten masked action probabilities without changing weights.

        Inputs:
            state: Current six-bit chemical state.
            mask: Legal textbook towers at that state.
        Returns:
            Ten probabilities ordered by external tower ID.
        """

        with torch.no_grad():
            distribution = self._distribution(state, mask)
            return tuple(float(value) for value in distribution.probs.tolist())

    def update(self, episodes: Iterable[EpisodeTrajectory]) -> float:
        """Take one Adam step using complete current-policy trajectories.

        Inputs:
            episodes: Nonempty batch of complete on-policy episodes.
        Returns:
            Scalar loss before the optimizer step.
        """

        batch = tuple(episodes)
        if not batch:
            raise ValueError("A policy update requires at least one episode.")

        # Average full-episode likelihood-ratio losses over the sampled batch.
        losses = []
        for episode in batch:
            _validate_trajectory(episode)
            returns = reward_to_go(step.reward for step in episode.steps)
            terms = [step.log_probability * value for step, value in zip(episode.steps, returns)]
            losses.append(-torch.stack(terms).sum())

        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().item())



