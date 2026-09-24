"""
Last modified time: 2026-09-24
Last modified content: Consolidate verified chemistry and remove redundant work
Last modified by: OpenAI Codex
File design: Single source of truth for textbook data
File purpose: Provide immutable constants and column specifications for the environment
File creator: OpenAI Codex
"""

from __future__ import annotations

from types import MappingProxyType

from distillation_sequencing_env.models import ColumnSpec, State


TOTAL_FEED_KMOL_PER_HOUR = 1000.0
HEATING_UTILITY_COST = 34.0
COOLING_UTILITY_COST = 1.3
TOTAL_UTILITY_COST = HEATING_UTILITY_COST + COOLING_UTILITY_COST

MIXTURE_ORDER = ("ABCD", "ABC", "BCD", "AB", "BC", "CD")
INTERMEDIATE_MIXTURES = frozenset(MIXTURE_ORDER)
INITIAL_STATE: State = (1, 0, 0, 0, 0, 0)
TERMINAL_STATE: State = (0, 0, 0, 0, 0, 0)

# Component mole fractions exactly follow the initial feed data in Table 17.1.
COMPONENT_MOLE_FRACTIONS = MappingProxyType({
    "A": 0.15,
    "B": 0.30,
    "C": 0.35,
    "D": 0.20,
})

# Specifications follow textbook action IDs; product names also define sharp-split transitions.
COLUMN_SPECS = (
    ColumnSpec(1, "A/BCD", "ABCD", "A", "BCD", 145.0, 0.42, 0.028),
    ColumnSpec(2, "AB/CD", "ABCD", "AB", "CD", 52.0, 0.12, 0.042),
    ColumnSpec(3, "ABC/D", "ABCD", "ABC", "D", 76.0, 0.25, 0.054),
    ColumnSpec(4, "B/CD", "BCD", "B", "CD", 38.0, 0.14, 0.040),
    ColumnSpec(5, "BC/D", "BCD", "BC", "D", 66.0, 0.21, 0.047),
    ColumnSpec(6, "A/BC", "ABC", "A", "BC", 125.0, 0.78, 0.024),
    ColumnSpec(7, "AB/C", "ABC", "AB", "C", 44.0, 0.11, 0.039),
    ColumnSpec(8, "C/D", "CD", "C", "D", 58.0, 0.19, 0.044),
    ColumnSpec(9, "B/C", "BC", "B", "C", 37.0, 0.08, 0.036),
    ColumnSpec(10, "A/B", "AB", "A", "B", 112.0, 0.39, 0.022),
)

COLUMN_BY_ACTION = MappingProxyType({column.action_id: column for column in COLUMN_SPECS})
STATE_SIZE = len(MIXTURE_ORDER)
ACTION_COUNT = len(COLUMN_SPECS)
EPISODE_STEPS = len(COMPONENT_MOLE_FRACTIONS) - 1
