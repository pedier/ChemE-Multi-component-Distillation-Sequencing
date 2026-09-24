"""
Last modified time: 2026-09-14-15:44
Last modified content: Finalize the stage-three public package interface
Last modified by: OpenAI Codex
File design: Package-level public entry point
File purpose: Export data, environment, Q-learning, training, and evaluation APIs
File creator: OpenAI Codex
"""

from distillation_q_learning.data import (
    COLUMN_SPECS,
    INITIAL_STATE,
    MIXTURE_ORDER,
    TERMINAL_STATE,
)
from distillation_q_learning.environment import (
    DistillationSequenceEnvironment,
    action_mask,
    active_mixtures,
    apply_action,
    available_actions,
    calculate_column_economics,
    encode_state,
)
from distillation_q_learning.models import (
    ActionMask,
    ColumnEconomics,
    ColumnSpec,
    EvaluationResult,
    State,
    StepInfo,
    StepResult,
)
from distillation_q_learning.q_learning import (
    QLearningConfig,
    TabularQLearningAgent,
    epsilon_for_episode,
)
from distillation_q_learning.training import (
    DEFAULT_EPISODES,
    EPISODE_STEPS,
    evaluate,
    train,
)


# Keep the public surface explicit so later learning modules can extend it safely.
__all__ = [
    "ActionMask",
    "COLUMN_SPECS",
    "ColumnEconomics",
    "ColumnSpec",
    "DistillationSequenceEnvironment",
    "DEFAULT_EPISODES",
    "EPISODE_STEPS",
    "EvaluationResult",
    "INITIAL_STATE",
    "MIXTURE_ORDER",
    "QLearningConfig",
    "State",
    "StepInfo",
    "StepResult",
    "TERMINAL_STATE",
    "TabularQLearningAgent",
    "action_mask",
    "active_mixtures",
    "apply_action",
    "available_actions",
    "calculate_column_economics",
    "encode_state",
    "epsilon_for_episode",
    "evaluate",
    "train",
]
