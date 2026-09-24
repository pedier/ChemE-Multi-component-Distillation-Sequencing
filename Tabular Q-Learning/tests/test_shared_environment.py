"""
Last modified time: 2026-09-24
Last modified content: Verify shared contracts, strict inputs and seeded cleanup behavior
Last modified by: OpenAI Codex
File design: Compatibility and package identity regression test
File purpose: Guard the single environment implementation used by this learner
File creator: OpenAI Codex
"""

from __future__ import annotations

from distillation_sequencing_env import (
    COLUMN_SPECS as shared_columns,
    DistillationSequenceEnvironment as SharedEnvironment,
    StepResult as SharedStepResult,
    EvaluationResult as SharedEvaluationResult,
    available_actions as shared_actions,
)
from distillation_q_learning import (
    COLUMN_SPECS as legacy_columns,
    DistillationSequenceEnvironment as LegacyEnvironment,
    StepResult as LegacyStepResult,
    EvaluationResult as LegacyEvaluationResult,
    available_actions as legacy_actions,
)
from distillation_q_learning.data import COLUMN_SPECS as module_columns
from distillation_q_learning.environment import DistillationSequenceEnvironment as ModuleEnvironment
from distillation_q_learning.models import StepResult as ModuleStepResult


def test_legacy_exports_are_the_shared_chemical_objects() -> None:
    """Verify old import paths expose the one shared model.

    Inputs:
        Public and module-level chemical exports from both packages.
    Returns:
        None. Identity assertions reject copied domain objects and logic.
    """

    assert legacy_columns is module_columns is shared_columns
    assert LegacyEnvironment is ModuleEnvironment is SharedEnvironment
    assert LegacyStepResult is ModuleStepResult is SharedStepResult
    assert legacy_actions is shared_actions
    assert LegacyEvaluationResult is SharedEvaluationResult
