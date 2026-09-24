"""
Last modified time: 2026-09-14-15:44
Last modified content: Reuse shared evaluation while preserving policy semantics
Last modified by: OpenAI Codex
File design: Standard-library command-line and serialization boundary
File purpose: Train, evaluate, and print a reproducible experiment as JSON
File creator: OpenAI Codex
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from distillation_sequencing_env.evaluation import reachable_states
from distillation_q_learning.models import EvaluationResult, State
from distillation_q_learning.q_learning import QLearningConfig, TabularQLearningAgent
from distillation_q_learning.training import DEFAULT_EPISODES, evaluate, train


def _positive_integer(value: str) -> int:
    """Parse one positive command-line integer.

    Inputs:
        value: Raw command-line token.
    Returns:
        Parsed positive integer.
    """

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _ordered_nonterminal_states() -> tuple[State, ...]:
    """Enumerate reachable nonterminal states in canonical decision order.

    Inputs:
        None.
    Returns:
        States sorted first by decision depth and then by state tuple.
    """


    return reachable_states(canonical=True)


def _policy_records(agent: TabularQLearningAgent) -> list[dict[str, Any]]:
    """Serialize the greedy policy over every reachable nonterminal state.

    Inputs:
        agent: Trained tabular Q-learning agent.
    Returns:
        Ordered JSON-compatible policy records.
    """

    return [
        {"state": list(state), "action_id": agent.greedy_action(state)}
        for state in _ordered_nonterminal_states()
    ]


def _q_value_records(agent: TabularQLearningAgent) -> list[dict[str, Any]]:
    """Serialize all legal Q values in deterministic state-action order.

    Inputs:
        agent: Trained tabular Q-learning agent.
    Returns:
        Ordered JSON-compatible Q-value records.
    """

    records: list[dict[str, Any]] = []
    for state in _ordered_nonterminal_states():
        for action_id, value in agent.q_values(state).items():
            records.append({"state": list(state), "action_id": action_id, "value": value})
    return records


def _evaluation_record(result: EvaluationResult) -> dict[str, Any]:
    """Convert an evaluation result into a JSON-compatible mapping.

    Inputs:
        result: Immutable greedy evaluation result.
    Returns:
        Mapping containing trajectory, flowsheet, cost, and reward fields.
    """

    # Tuples become JSON arrays without changing the numerical experiment values.
    return {
        "final_state": list(result.final_state),
        "action_sequence": list(result.action_sequence),
        "normalized_column_set": list(result.normalized_column_set),
        "selection_vector": list(result.selection_vector),
        "total_cost_musd_per_year": result.total_cost_musd_per_year,
        "total_reward": result.total_reward,
    }


def build_experiment_payload(config: QLearningConfig, episodes: int) -> dict[str, Any]:
    """Train and serialize one reproducible tabular Q-learning experiment.

    Inputs:
        config: Validated Q-learning configuration.
        episodes: Positive number of training episodes.
    Returns:
        JSON-compatible experiment payload with policy, Q values, and evaluation.
    """

    agent = train(config, episodes)
    result = evaluate(agent)

    # The configuration record makes every source of training variability explicit.
    config_record = {
        "episodes": episodes,
        "learning_rate": config.learning_rate,
        "discount_factor": config.discount_factor,
        "epsilon_start": config.epsilon_start,
        "epsilon_min": config.epsilon_min,
        "epsilon_decay": config.epsilon_decay,
        "seed": config.seed,
    }

    # Record order is canonical and dictionary keys are sorted when JSON is printed.
    return {
        "config": config_record,
        "policy": _policy_records(agent),
        "q_values": _q_value_records(agent),
        "evaluation": _evaluation_record(result),
    }


def _argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Inputs:
        None.
    Returns:
        Parser supporting episode-count and random-seed overrides.
    """

    parser = argparse.ArgumentParser(description="Train the distillation Q-learning agent.")
    parser.add_argument("--episodes", type=_positive_integer, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=QLearningConfig().seed)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run training and print a deterministic JSON document.

    Inputs:
        argv: Optional argument sequence; process arguments are used when omitted.
    Returns:
        Zero process status after successful serialization.
    """

    arguments = _argument_parser().parse_args(argv)
    config = QLearningConfig(seed=arguments.seed)
    payload = build_experiment_payload(config, arguments.episodes)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
