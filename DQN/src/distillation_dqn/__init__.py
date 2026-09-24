"""
Last modified time: 2026-09-21-03:06
Last modified content: Export complete environment, DQN, training, and evaluation APIs
Last modified by: OpenAI Codex
File design: Explicit package-level public interface
File purpose: Export textbook data and reproducible DQN experiment operations
File creator: OpenAI Codex
"""

from distillation_dqn.agent import DQNConfig, DuelingDoubleDQNAgent
from distillation_dqn.data import COLUMN_SPECS, INITIAL_STATE, MIXTURE_ORDER, TERMINAL_STATE
from distillation_dqn.environment import (
    DistillationSequenceEnvironment,
    action_mask,
    active_mixtures,
    apply_action,
    available_actions,
    calculate_column_economics,
    encode_state,
)
from distillation_dqn.models import (
    ActionMask,
    ColumnEconomics,
    ColumnSpec,
    EvaluationResult,
    State,
    StepInfo,
    StepResult,
)
from distillation_dqn.network import DuelingQNetwork
from distillation_dqn.replay import ReplayBuffer, Transition
from distillation_dqn.targets import double_dqn_targets
from distillation_dqn.training import (
    DEFAULT_EPISODES,
    EPISODE_STEPS,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    epsilon_for_episode,
    evaluate,
    train,
)


__all__ = [
    "ActionMask",
    "COLUMN_SPECS",
    "ColumnEconomics",
    "ColumnSpec",
    "DEFAULT_EPISODES",
    "DQNConfig",
    "DistillationSequenceEnvironment",
    "DuelingDoubleDQNAgent",
    "DuelingQNetwork",
    "EPISODE_STEPS",
    "EPSILON_DECAY",
    "EPSILON_MIN",
    "EPSILON_START",
    "EvaluationResult",
    "INITIAL_STATE",
    "MIXTURE_ORDER",
    "ReplayBuffer",
    "State",
    "StepInfo",
    "StepResult",
    "TERMINAL_STATE",
    "Transition",
    "action_mask",
    "active_mixtures",
    "apply_action",
    "available_actions",
    "calculate_column_economics",
    "double_dqn_targets",
    "encode_state",
    "epsilon_for_episode",
    "evaluate",
    "train",
]
