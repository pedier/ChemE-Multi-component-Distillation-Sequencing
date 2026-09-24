"""
Last modified time: 2026-09-24
Last modified content: Reuse shared evaluation while preserving policy semantics
Last modified by: OpenAI Codex
File design: Immutable rollout records plus a small actor-critic learning agent
File purpose: Optimize feasible distillation policies without changing chemical rewards
File creator: OpenAI Codex
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.distributions import Categorical

from distillation_sequencing_env.data import ACTION_COUNT, EPISODE_STEPS, INITIAL_STATE, TERMINAL_STATE
from distillation_sequencing_env.environment import action_mask, apply_action, validate_reward
from distillation_ppo.models import ActionMask, State
from distillation_ppo.rollout import (
    ActionSample, RolloutStep, EpisodeTrajectory, PreparedBatch, PPOUpdateStats,
)
from distillation_ppo.policy import ActorNetwork, CriticNetwork, masked_distribution


@dataclass(frozen=True)
class PPOConfig:
    """Hold fixed actor-critic optimization and reproducibility settings.

    Inputs:
        learning_rate: Adam step size.
        batch_episodes: Complete episodes collected before each update.
        update_epochs: Passes over one fresh on-policy batch.
        minibatch_size: Transitions used in each gradient step.
        clip_epsilon: PPO probability-ratio clipping radius.
        gae_lambda: GAE trace parameter; economic discount remains one.
        value_coefficient: Weight of the critic mean-squared error.
        entropy_coefficient: Training-only exploration regularizer.
        max_gradient_norm: Maximum actor-critic gradient norm.
        seed: PyTorch initialization and sampling seed.
    Returns:
        Immutable validated PPO settings.
    """

    learning_rate: float = 3e-4
    batch_episodes: int = 32
    update_epochs: int = 4
    minibatch_size: int = 48
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        """Reject invalid settings before training can start.

        Inputs:
            None. Reads this configuration's fields.
        Returns:
            None. Raises ValueError for an invalid setting.
        """

        # Numeric validation keeps optimization parameters finite and correctly bounded.
        positive = ("learning_rate", "clip_epsilon", "max_gradient_norm")
        nonnegative = ("value_coefficient", "entropy_coefficient", "gae_lambda")
        for name in (*positive, *nonnegative):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite numeric value.")
            if not math.isfinite(value) or (name in positive and value <= 0):
                raise ValueError(f"{name} is outside its valid range.")
            if name in nonnegative and value < 0:
                raise ValueError(f"{name} cannot be negative.")

        if self.clip_epsilon >= 1 or self.gae_lambda > 1:
            raise ValueError("clip_epsilon must be below one and gae_lambda at most one.")

        for name in ("batch_episodes", "update_epochs", "minibatch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")


def validate_episode(episode: EpisodeTrajectory, policy_version: int) -> None:
    """Check physical validity and freshness of a complete PPO episode.

    Inputs:
        episode: Candidate sequence of sampled chemical transitions.
        policy_version: Current agent version expected for all samples.
    Returns:
        None. Raises ValueError if the episode is invalid or stale.
    """

    if len(episode.steps) != EPISODE_STEPS or episode.steps[0].state != INITIAL_STATE:
        raise ValueError("PPO requires a complete three-step episode from ABCD.")

    expected_state = INITIAL_STATE
    # Every recorded action must match the known sharp-split model and tower cost.
    for index, step in enumerate(episode.steps):
        sample = step.sample
        if step.state != expected_state or step.mask != action_mask(step.state):
            raise ValueError("Trajectory state or action mask is inconsistent.")
        if (sample.policy_version != policy_version or type(sample.action_id) is not int
                or sample.action_id not in range(1, ACTION_COUNT + 1)
                or not step.mask[sample.action_id - 1]):
            raise ValueError("Trajectory contains a stale or infeasible action.")
        if not all(math.isfinite(value) for value in (
            step.reward, sample.old_log_probability, sample.old_value
        )):
            raise ValueError("Trajectory contains nonfinite rollout statistics.")

        validate_reward(sample.action_id, step.reward)
        expected_state = apply_action(step.state, sample.action_id)
        if step.next_state != expected_state:
            raise ValueError("Trajectory reward or deterministic transition is incorrect.")
        if step.terminated != (expected_state == TERMINAL_STATE) or step.terminated != (index == EPISODE_STEPS - 1):
            raise ValueError("Trajectory has an invalid terminal boundary.")


def gae_advantages(
    rewards: Iterable[float], values: Iterable[float], terminated: Iterable[bool],
    gae_lambda: float = 0.95,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compute undiscounted GAE advantages and critic targets for one episode.

    Inputs:
        rewards: Ordered negative tower costs.
        values: Behavior-critic estimates at the same states.
        terminated: Terminal flags, true only after the final transition.
        gae_lambda: Trace weight in the inclusive interval zero to one.
    Returns:
        Ordered advantages and value targets, each with one entry per action.
    """

    reward_values = tuple(rewards)
    old_values = tuple(values)
    done_values = tuple(terminated)
    if not reward_values or len(reward_values) != len(old_values) or len(reward_values) != len(done_values):
        raise ValueError("GAE inputs must be nonempty and have equal lengths.")
    if any(done_values[:-1]) or done_values[-1] is not True:
        raise ValueError("GAE requires a complete episode with one final terminal step.")
    if any(not math.isfinite(value) for value in (*reward_values, *old_values)):
        raise ValueError("GAE rewards and values must be finite.")
    if not math.isfinite(gae_lambda) or not 0 <= gae_lambda <= 1:
        raise ValueError("gae_lambda must lie between zero and one.")

    # Backward recursion uses gamma one and zero bootstrap at termination.
    advantages = [0.0] * len(reward_values)
    running = 0.0
    for index in range(len(reward_values) - 1, -1, -1):
        continuation = 0.0 if done_values[index] else old_values[index + 1]
        delta = reward_values[index] + continuation - old_values[index]
        running = delta + (0.0 if done_values[index] else gae_lambda * running)
        advantages[index] = running

    returns = tuple(advantage + value for advantage, value in zip(advantages, old_values))
    return tuple(advantages), returns


def clipped_policy_loss(
    new_log_probabilities: Tensor, old_log_probabilities: Tensor,
    advantages: Tensor, clip_epsilon: float,
) -> Tensor:
    """Compute the negative PPO clipped surrogate objective.

    Inputs:
        new_log_probabilities: Current actor log probabilities for sampled actions.
        old_log_probabilities: Frozen behavior-policy log probabilities.
        advantages: Fixed GAE advantages for the sampled actions.
        clip_epsilon: Allowed probability-ratio deviation from one.
    Returns:
        Scalar differentiable actor loss to minimize.
    """

    if (new_log_probabilities.ndim != 1 or not new_log_probabilities.numel()
            or new_log_probabilities.shape != old_log_probabilities.shape
            or new_log_probabilities.shape != advantages.shape):
        raise ValueError("PPO loss inputs must be matching nonempty vectors.")
    if not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must lie strictly between zero and one.")

    ratio = torch.exp(new_log_probabilities - old_log_probabilities)
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    return -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()


class PPOAgent:
    """Sample legal towers and optimize an actor-critic with clipped PPO.

    Inputs:
        config: Validated learning and random-seed settings.
    Returns:
        Actor, critic, and optimizer for the fixed chemical environment.
    """

    def __init__(self, config: PPOConfig | None = None) -> None:
        """Initialize seeded CPU networks and a joint Adam optimizer.

        Inputs:
            config: Optional validated PPO configuration.
        Returns:
            None. Stores actor, critic, optimizer, and policy version.
        """

        # Initialize both networks under one seed and optimize their joint objective.
        self.config = config or PPOConfig()
        torch.manual_seed(self.config.seed)
        self.actor = ActorNetwork().cpu()
        self.critic = CriticNetwork().cpu()
        parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate)
        self.policy_version = 0
        self.training_returns: list[float] = []
        self.update_history: list[PPOUpdateStats] = []

    def _distribution(self, state: State, mask: ActionMask) -> Categorical:
        """Build a policy distribution after verifying the chemical mask.

        Inputs:
            state: Current six-bit chemical state.
            mask: Claimed feasible tower mask for this state.
        Returns:
            Categorical distribution over only feasible towers.
        """

        if mask != action_mask(state):
            raise ValueError("Action mask does not match the chemical state.")

        state_tensor = torch.tensor(state, dtype=torch.float32)
        return masked_distribution(self.actor(state_tensor), mask)

    def sample_action(self, state: State, mask: ActionMask) -> ActionSample:
        """Sample one feasible tower while freezing old policy statistics.

        Inputs:
            state: Current six-bit chemical state.
            mask: Verified feasible tower mask.
        Returns:
            External tower ID, old log probability, old value, and policy version.
        """

        if mask != action_mask(state):
            raise ValueError("Action mask does not match the chemical state.")

        # Freeze sampling statistics before any later policy update.
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            distribution = masked_distribution(self.actor(state_tensor), mask)
            action_index = distribution.sample()
            value = self.critic(state_tensor)
            return ActionSample(
                int(action_index.item()) + 1,
                float(distribution.log_prob(action_index).item()),
                float(value.item()),
                self.policy_version,
            )

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
        """Choose the most probable legal tower with smallest-ID tie breaking.

        Inputs:
            state: Current six-bit chemical state.
            mask: Verified feasible tower mask.
        Returns:
            External tower ID for a deterministic pure policy.
        """

        with torch.no_grad():
            distribution = self._distribution(state, mask)
            return int(torch.argmax(distribution.logits).item()) + 1

    def action_probabilities(self, state: State, mask: ActionMask) -> tuple[float, ...]:
        """Report ten tower probabilities without changing the agent.

        Inputs:
            state: Current six-bit chemical state.
            mask: Verified feasible tower mask.
        Returns:
            Ten probabilities in external tower-ID order.
        """

        with torch.no_grad():
            distribution = self._distribution(state, mask)
            return tuple(float(probability) for probability in distribution.probs.tolist())

    def _prepare_batch(self, episodes: tuple[EpisodeTrajectory, ...]) -> PreparedBatch:
        """Validate fresh episodes and freeze GAE targets as tensors.

        Inputs:
            episodes: Nonempty batch collected under this policy version.
        Returns:
            Tensor group for repeated PPO minibatch updates.
        """

        if not episodes:
            raise ValueError("PPO needs at least one fresh complete episode.")

        steps: list[RolloutStep] = []
        advantages: list[float] = []
        returns: list[float] = []
        # Compute each episode separately so terminal GAE cannot cross a reset.
        for episode in episodes:
            validate_episode(episode, self.policy_version)
            episode_advantages, episode_returns = gae_advantages(
                (step.reward for step in episode.steps),
                (step.sample.old_value for step in episode.steps),
                (step.terminated for step in episode.steps),
                self.config.gae_lambda,
            )
            steps.extend(episode.steps)
            advantages.extend(episode_advantages)
            returns.extend(episode_returns)

        advantage_tensor = torch.tensor(advantages, dtype=torch.float32)
        standard_deviation = advantage_tensor.std(unbiased=False)
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / standard_deviation.clamp_min(1e-8)
        return PreparedBatch(
            torch.tensor([step.state for step in steps], dtype=torch.float32),
            torch.tensor([step.mask for step in steps], dtype=torch.bool),
            torch.tensor([step.sample.action_id - 1 for step in steps], dtype=torch.long),
            torch.tensor([step.sample.old_log_probability for step in steps], dtype=torch.float32),
            advantage_tensor,
            torch.tensor(returns, dtype=torch.float32),
        )

    def _loss_terms(self, batch: PreparedBatch, indices: Tensor) -> tuple[Tensor, ...]:
        """Compute actor, critic, and diagnostic terms for one minibatch.

        Inputs:
            batch: Frozen on-policy tensors.
            indices: Selected transition indices for this gradient step.
        Returns:
            Actor loss, value loss, entropy, approximate KL, and clip fraction.
        """

        # Evaluate the clipped actor, critic, and diagnostics on the same legal minibatch.
        distribution = masked_distribution(self.actor(batch.states[indices]), batch.masks[indices])
        new_log_probability = distribution.log_prob(batch.action_indices[indices])
        old_log_probability = batch.old_log_probabilities[indices]
        actor_loss = clipped_policy_loss(
            new_log_probability, old_log_probability,
            batch.advantages[indices], self.config.clip_epsilon,
        )
        value_loss = torch.mean((self.critic(batch.states[indices]) - batch.returns[indices]) ** 2)
        entropy = distribution.entropy().mean()
        log_ratio = new_log_probability - old_log_probability
        ratio = torch.exp(log_ratio)
        approximate_kl = (ratio - 1 - log_ratio).mean()
        clip_fraction = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean()
        return actor_loss, value_loss, entropy, approximate_kl, clip_fraction

    def update(self, episodes: Iterable[EpisodeTrajectory]) -> PPOUpdateStats:
        """Reuse one fresh rollout batch for clipped PPO minibatch updates.

        Inputs:
            episodes: Complete trajectories sampled under the current version.
        Returns:
            Mean optimization diagnostics; increments the policy version once.
        """

        batch = self._prepare_batch(tuple(episodes))
        metrics: list[tuple[float, ...]] = []
        sample_count = batch.states.shape[0]
        parameters = list(self.actor.parameters()) + list(self.critic.parameters())

        # Reuse this frozen on-policy batch for configured epochs, then expire it.
        for _ in range(self.config.update_epochs):
            order = torch.randperm(sample_count)
            for offset in range(0, sample_count, self.config.minibatch_size):
                indices = order[offset:offset + self.config.minibatch_size]
                self.optimizer.zero_grad()
                terms = self._loss_terms(batch, indices)
                loss = terms[0] + self.config.value_coefficient * terms[1]
                loss -= self.config.entropy_coefficient * terms[2]
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("PPO loss became nonfinite.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, self.config.max_gradient_norm)
                self.optimizer.step()
                metrics.append(tuple(float(term.detach().item()) for term in terms))

        self.policy_version += 1
        averages = tuple(sum(row[index] for row in metrics) / len(metrics) for index in range(5))
        return PPOUpdateStats(*averages, sample_count, len(metrics))
