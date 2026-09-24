"""
Last modified time: 2026-09-21-03:05
Last modified content: Reuse shared evaluation while preserving policy semantics
Last modified by: OpenAI Codex
File design: Standard-library command-line and JSON boundary
File purpose: Train, evaluate, and report the fixed DQN experiment
File creator: OpenAI Codex
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from distillation_dqn.agent import DQNConfig, DuelingDoubleDQNAgent
from distillation_sequencing_env.evaluation import reachable_states
from distillation_dqn.models import EvaluationResult, State
from distillation_dqn.training import (
    DEFAULT_EPISODES,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    evaluate,
    train,
)


def _positive_integer(value: str) -> int:
    """Parse one strictly positive command-line integer.

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
    """Enumerate reachable decision states in canonical breadth-first order.

    Inputs:
        None.
    Returns:
        States sorted by decision depth and then their binary tuple.
    """


    return reachable_states(canonical=True)


def _policy_records(agent: DuelingDoubleDQNAgent) -> list[dict[str, Any]]:
    """Serialize the pure policy across reachable nonterminal states.

    Inputs:
        agent: Trained DQN agent.
    Returns:
        Canonically ordered JSON-compatible state-action records.
    """

    return [
        {"state": list(state), "action_id": agent.greedy_action(state)}
        for state in _ordered_nonterminal_states()
    ]


def _q_value_records(agent: DuelingDoubleDQNAgent) -> list[dict[str, Any]]:
    """Serialize all legal network Q estimates in deterministic order.

    Inputs:
        agent: Trained DQN agent.
    Returns:
        Ordered records containing state, tower ID, and estimated Q value.
    """

    records: list[dict[str, Any]] = []
    for state in _ordered_nonterminal_states():
        for action_id, value in agent.q_values(state).items():
            records.append({"state": list(state), "action_id": action_id, "value": value})
    return records


def _evaluation_record(result: EvaluationResult) -> dict[str, Any]:
    """Convert one greedy evaluation into JSON-compatible fields.

    Inputs:
        result: Immutable economic and trajectory evaluation result.
    Returns:
        Mapping of final state, towers, selection vector, cost, and reward.
    """

    # Keep the execution order distinct from the normalized tower set.
    return {
        "final_state": list(result.final_state),
        "action_sequence": list(result.action_sequence),
        "normalized_column_set": list(result.normalized_column_set),
        "selection_vector": list(result.selection_vector),
        "total_cost_musd_per_year": result.total_cost_musd_per_year,
        "total_reward": result.total_reward,
    }


def build_experiment_payload(config: DQNConfig, episodes: int) -> dict[str, Any]:
    """Train once and assemble the complete reproducible experiment record.

    Inputs:
        config: Validated DQN learning configuration.
        episodes: Positive episode count.
    Returns:
        JSON-compatible config, policy, Q values, and evaluation records.
    """

    agent = train(config, episodes)
    result = evaluate(agent)
    # Record every fixed and configurable source of experiment variation.
    config_record = {
        **asdict(config),
        "episodes": episodes,
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": EPSILON_DECAY,
        "network": "6-64-64-dueling",
        "device": "cpu",
    }

    return {
        "config": config_record,
        "policy": _policy_records(agent),
        "q_values": _q_value_records(agent),
        "evaluation": _evaluation_record(result),
    }


def _argument_parser() -> argparse.ArgumentParser:
    """Construct the episode-count and random-seed CLI parser.

    Inputs:
        None.
    Returns:
        Argument parser for one DQN experiment.
    """

    parser = argparse.ArgumentParser(description="Train the masked distillation DQN agent.")
    parser.add_argument("--episodes", type=_positive_integer, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DQNConfig().seed)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one DQN experiment and print stable JSON to standard output.

    Inputs:
        argv: Optional arguments; process arguments are used if omitted.
    Returns:
        Zero exit status after a successful experiment.
    """

    arguments = _argument_parser().parse_args(argv)
    payload = build_experiment_payload(DQNConfig(seed=arguments.seed), arguments.episodes)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0

