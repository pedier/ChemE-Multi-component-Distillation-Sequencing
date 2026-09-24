"""
Last modified time: 2026-09-24
Last modified content: Forward the chemical data to the shared package
Last modified by: OpenAI Codex
File design: Compatibility exports for the established distillation_dqn import path
File purpose: Preserve callers while all learners use one chemical environment
File creator: OpenAI Codex
"""

from distillation_sequencing_env.data import (
    COLUMN_BY_ACTION,
    COLUMN_SPECS,
    COMPONENT_MOLE_FRACTIONS,
    COOLING_UTILITY_COST,
    HEATING_UTILITY_COST,
    INITIAL_STATE,
    INTERMEDIATE_MIXTURES,
    MIXTURE_ORDER,
    TERMINAL_STATE,
    TOTAL_FEED_KMOL_PER_HOUR,
    TOTAL_UTILITY_COST,
)
