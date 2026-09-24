"""
Last modified time: 2026-09-24
Last modified content: Forward the chemical operations to the shared package
Last modified by: OpenAI Codex
File design: Compatibility exports for the established distillation_dqn import path
File purpose: Preserve callers while all learners use one chemical environment
File creator: OpenAI Codex
"""

from distillation_sequencing_env.environment import (
    DistillationSequenceEnvironment,
    action_mask,
    active_mixtures,
    apply_action,
    available_actions,
    calculate_column_economics,
    encode_state,
)
