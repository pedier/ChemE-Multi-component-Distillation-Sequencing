"""
Last modified time: 2026-09-24
Last modified content: Consolidate verified chemistry and remove redundant work
Last modified by: OpenAI Codex
File design: Immutable domain data models
File purpose: Define typed contracts for textbook data, states, and step results
File creator: OpenAI Codex
"""

from __future__ import annotations

from dataclasses import dataclass


State = tuple[int, int, int, int, int, int]
ActionMask = tuple[bool, bool, bool, bool, bool, bool, bool, bool, bool, bool]


@dataclass(frozen=True)
class ColumnSpec:
    """Describe one candidate distillation column from the textbook.

    Inputs:
        Fields supplied by the data module: action ID, streams, and economic parameters.
    Returns:
        An immutable candidate-column specification.
    """

    action_id: int
    separator: str
    feed_mixture: str
    light_product: str
    heavy_product: str
    fixed_cost_kusd_per_year: float
    variable_cost_factor: float
    heat_duty_coefficient: float


@dataclass(frozen=True)
class ColumnEconomics:
    """Store flow, heat duty, and cost produced by one column choice.

    Inputs:
        Values calculated from the column specification and initial feed composition.
    Returns:
        An immutable economics record using the textbook scaling conventions.
    """

    feed_flow_kmol_per_hour: float
    heat_duty_q: float
    annual_cost_kusd_per_year: float
    annual_cost_musd_per_year: float


@dataclass(frozen=True)
class StepInfo:
    """Record engineering information after one environment action.

    Inputs:
        The selected column, economics, cumulative cost, and action history.
    Returns:
        An immutable information record for auditing and reporting.
    """

    action_id: int
    feed_mixture: str
    economics: ColumnEconomics
    cumulative_cost_musd_per_year: float
    selected_actions: tuple[int, ...]
    selection_vector: tuple[int, ...]


@dataclass(frozen=True)
class StepResult:
    """Describe the complete result of one environment transition.

    Inputs:
        The next state, reward, termination flag, action mask, and engineering details.
    Returns:
        An immutable step result for use by an agent.
    """

    state: State
    reward: float
    terminated: bool
    action_mask: ActionMask
    info: StepInfo


@dataclass(frozen=True)
class EvaluationResult:
    """Store the result of one deterministic greedy evaluation.

    Inputs:
        Final state, selected columns, selection vector, cost, and total reward.
    Returns:
        An immutable record of the evaluated pure distillation strategy.
    """

    final_state: State
    action_sequence: tuple[int, ...]
    normalized_column_set: tuple[int, ...]
    selection_vector: tuple[int, ...]
    total_cost_musd_per_year: float
    total_reward: float
