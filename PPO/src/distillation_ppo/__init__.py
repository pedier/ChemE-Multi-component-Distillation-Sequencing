"""
Last modified time: 2026-09-24
Last modified content: Export the complete chemical environment and PPO workflow
Last modified by: OpenAI Codex
File design: Public package entry point
File purpose: Expose textbook chemistry, PPO learning, training, and evaluation
File creator: OpenAI Codex
"""

from distillation_ppo.data import COLUMN_SPECS, INITIAL_STATE, MIXTURE_ORDER, TERMINAL_STATE
from distillation_ppo.environment import (
    DistillationSequenceEnvironment,
    action_mask,
    active_mixtures,
    apply_action,
    available_actions,
    calculate_column_economics,
    encode_state,
)
from distillation_ppo.models import (
    ColumnEconomics,
    ColumnSpec,
    EvaluationResult,
    StepInfo,
    StepResult,
)
from distillation_ppo.policy import ActorNetwork, CriticNetwork, masked_distribution
from distillation_ppo.ppo import (
    ActionSample,
    EpisodeTrajectory,
    PPOAgent,
    PPOConfig,
    PPOUpdateStats,
    RolloutStep,
    clipped_policy_loss,
    gae_advantages,
    validate_episode,
)
from distillation_ppo.training import (
    evaluate,
    policy_snapshot,
    reachable_states,
    train,
    training_summary,
)

__all__ = [
    "COLUMN_SPECS",
    "INITIAL_STATE",
    "MIXTURE_ORDER",
    "TERMINAL_STATE",
    "ActionSample",
    "ActorNetwork",
    "ColumnEconomics",
    "ColumnSpec",
    "CriticNetwork",
    "DistillationSequenceEnvironment",
    "EpisodeTrajectory",
    "EvaluationResult",
    "PPOAgent",
    "PPOConfig",
    "PPOUpdateStats",
    "RolloutStep",
    "StepInfo",
    "StepResult",
    "action_mask",
    "active_mixtures",
    "apply_action",
    "available_actions",
    "calculate_column_economics",
    "clipped_policy_loss",
    "encode_state",
    "evaluate",
    "gae_advantages",
    "masked_distribution",
    "policy_snapshot",
    "reachable_states",
    "train",
    "training_summary",
    "validate_episode",
]