"""
Last modified time: 2026-09-24
Last modified content: Share deterministic graph traversal and policy evaluation
Last modified by: OpenAI Codex
File design: Immutable PPO rollout records
File purpose: Define behavior samples, prepared tensors, and update statistics
File creator: OpenAI Codex
"""

from dataclasses import dataclass
from torch import Tensor
from distillation_ppo.models import ActionMask, State


@dataclass(frozen=True)
class ActionSample:
    """Freeze one decision under the current masked behavior policy.

    Inputs:
        action_id: Sampled external tower ID.
        old_log_probability: Log probability before any PPO update.
        old_value: Critic estimate before any PPO update.
        policy_version: Agent update count at the time of sampling.
    Returns:
        Immutable action sample for an on-policy rollout.
    """

    action_id: int
    old_log_probability: float
    old_value: float
    policy_version: int


@dataclass(frozen=True)
class RolloutStep:
    """Record one physical transition and its behavior-policy sample.

    Inputs:
        state: Six-bit state before the tower choice.
        mask: Legal tower mask at that state.
        sample: Action and frozen behavior-policy statistics.
        reward: Negative incremental annualized tower cost.
        next_state: State produced by the selected sharp split.
        terminated: Whether all four products are pure afterward.
    Returns:
        Immutable transition for GAE and PPO.
    """

    state: State
    mask: ActionMask
    sample: ActionSample
    reward: float
    next_state: State
    terminated: bool


@dataclass(frozen=True)
class EpisodeTrajectory:
    """Store one complete three-column on-policy trajectory.

    Inputs:
        steps: Ordered physical transitions from ABCD to pure products.
    Returns:
        Immutable episode record used once by PPO.
    """

    steps: tuple[RolloutStep, ...]


@dataclass(frozen=True)
class PreparedBatch:
    """Hold tensors derived from a validated, fresh rollout batch.

    Inputs:
        states: Chemical states sampled under one policy version.
        masks: Legal actions for the sampled states.
        action_indices: Zero-based sampled tower indices.
        old_log_probabilities: Frozen behavior-policy log probabilities.
        advantages: Batch-normalized GAE advantages.
        returns: Critic regression targets from unnormalized GAE.
    Returns:
        Immutable tensor group for minibatch optimization.
    """

    states: Tensor
    masks: Tensor
    action_indices: Tensor
    old_log_probabilities: Tensor
    advantages: Tensor
    returns: Tensor


@dataclass(frozen=True)
class PPOUpdateStats:
    """Summarize one completed update on fresh rollout data.

    Inputs:
        actor_loss: Mean clipped actor loss over all minibatches.
        value_loss: Mean critic mean-squared error.
        entropy: Mean legal-action policy entropy.
        approximate_kl: Mean nonnegative ratio-based KL estimate.
        clip_fraction: Fraction of sampled ratios outside the clip interval.
        transitions: Number of sampled transitions consumed.
        minibatches: Number of optimizer steps.
    Returns:
        Immutable diagnostics for reporting without changing environment reward.
    """

    actor_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    transitions: int
    minibatches: int
