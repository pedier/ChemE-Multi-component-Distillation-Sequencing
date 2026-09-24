"""
Last modified time: 2026-09-24
Last modified content: Translate metadata, docstrings, and code comments into English
Last modified by: OpenAI Codex
File design: Textbook data unit tests
File purpose: Prevent drift in Table 17.1 parameters and immutable domain models
File creator: OpenAI Codex
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from distillation_sequencing_env import COLUMN_SPECS, ColumnSpec
from distillation_sequencing_env.data import (
    COMPONENT_MOLE_FRACTIONS,
    TOTAL_FEED_KMOL_PER_HOUR,
    TOTAL_UTILITY_COST,
)


def test_feed_and_utility_constants_match_textbook() -> None:
    """Verify the feed and utility data supplied by the textbook.

    Inputs:
        None. Reads constants from the data module.
    Returns:
        None. The test passes when every textbook constant matches.
    """

    assert TOTAL_FEED_KMOL_PER_HOUR == 1000.0
    assert COMPONENT_MOLE_FRACTIONS == {"A": 0.15, "B": 0.30, "C": 0.35, "D": 0.20}
    assert sum(COMPONENT_MOLE_FRACTIONS.values()) == pytest.approx(1.0)
    assert TOTAL_UTILITY_COST == pytest.approx(35.3)


def test_column_specs_match_table_17_1() -> None:
    """Verify the separation task and three economic parameters for all columns.

    Inputs:
        None. Reads COLUMN_SPECS.
    Returns:
        None. The test passes when every specification matches Table 17.1.
    """

    # Expected values follow action IDs to verify order, connectivity, and economics together.
    expected = [
        (1, "A/BCD", "ABCD", "A", "BCD", 145.0, 0.42, 0.028),
        (2, "AB/CD", "ABCD", "AB", "CD", 52.0, 0.12, 0.042),
        (3, "ABC/D", "ABCD", "ABC", "D", 76.0, 0.25, 0.054),
        (4, "B/CD", "BCD", "B", "CD", 38.0, 0.14, 0.040),
        (5, "BC/D", "BCD", "BC", "D", 66.0, 0.21, 0.047),
        (6, "A/BC", "ABC", "A", "BC", 125.0, 0.78, 0.024),
        (7, "AB/C", "ABC", "AB", "C", 44.0, 0.11, 0.039),
        (8, "C/D", "CD", "C", "D", 58.0, 0.19, 0.044),
        (9, "B/C", "BC", "B", "C", 37.0, 0.08, 0.036),
        (10, "A/B", "AB", "A", "B", 112.0, 0.39, 0.022),
    ]
    actual = [tuple(column.__dict__.values()) for column in COLUMN_SPECS]
    assert actual == expected


def test_column_spec_is_immutable() -> None:
    """Verify that a column specification cannot be changed at runtime.

    Inputs:
        The first textbook column specification.
    Returns:
        None. The test passes when assignment raises FrozenInstanceError.
    """

    with pytest.raises(FrozenInstanceError):
        COLUMN_SPECS[0].action_id = 99  # type: ignore[misc]

    assert isinstance(COLUMN_SPECS[0], ColumnSpec)
