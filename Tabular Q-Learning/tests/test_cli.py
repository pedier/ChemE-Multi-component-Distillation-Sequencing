"""
Last modified time: 2026-09-14-15:43
Last modified content: Test deterministic experiment serialization and CLI behavior
Last modified by: OpenAI Codex
File design: Unit and integration tests for the JSON command-line boundary
File purpose: Verify canonical policy, Q-value, argument, and output records
File creator: OpenAI Codex
"""

from __future__ import annotations

import json

import pytest

from distillation_q_learning import INITIAL_STATE, QLearningConfig
from distillation_q_learning.cli import (
    _argument_parser,
    _ordered_nonterminal_states,
    _positive_integer,
    build_experiment_payload,
    main,
)


@pytest.fixture(scope="module")
def experiment_payload() -> dict[str, object]:
    """Build one accepted experiment payload for serialization assertions.

    Inputs:
        None.
    Returns:
        Payload trained for 10,000 episodes with seed 42.
    """

    return build_experiment_payload(QLearningConfig(seed=42), episodes=10_000)


def test_positive_integer_parser_accepts_only_positive_values() -> None:
    """Verify positive CLI integer conversion and rejection.

    Inputs:
        Valid and invalid raw command-line tokens.
    Returns:
        None. The test passes when only positive integers are accepted.
    """

    assert _positive_integer("12") == 12
    with pytest.raises(argparse_error_type(), match="positive"):
        _positive_integer("0")


def argparse_error_type() -> type[Exception]:
    """Return the argparse error class without duplicating a module import.

    Inputs:
        None.
    Returns:
        ArgumentTypeError class used by the positive integer parser.
    """

    import argparse

    return argparse.ArgumentTypeError


def test_argument_parser_supports_episode_and_seed_overrides() -> None:
    """Verify parsing of the two documented command-line options.

    Inputs:
        Explicit episode and seed tokens.
    Returns:
        None. The test passes when both override values are retained.
    """

    arguments = _argument_parser().parse_args(["--episodes", "25", "--seed", "7"])
    assert arguments.episodes == 25
    assert arguments.seed == 7

    with pytest.raises(SystemExit):
        _argument_parser().parse_args(["--episodes", "0"])


def test_reachable_states_use_canonical_depth_then_tuple_order() -> None:
    """Verify complete deterministic ordering of reachable policy states.

    Inputs:
        The exact environment transition model.
    Returns:
        None. The test passes when seven states appear in canonical order.
    """

    states = _ordered_nonterminal_states()
    assert len(states) == 7
    assert states[0] == INITIAL_STATE
    assert states[1:] == tuple(sorted(states[1:4])) + tuple(sorted(states[4:]))


def test_payload_has_complete_policy_q_values_and_evaluation(experiment_payload) -> None:
    """Verify the fixed JSON schema and all final acceptance values.

    Inputs:
        experiment_payload: Accepted deterministic experiment document.
    Returns:
        None. The test passes when schema, counts, and optimum match the plan.
    """

    payload = experiment_payload
    assert set(payload) == {"config", "policy", "q_values", "evaluation"}
    assert len(payload["policy"]) == 7  # type: ignore[arg-type]
    assert len(payload["q_values"]) == 12  # type: ignore[arg-type]

    evaluation = payload["evaluation"]
    assert evaluation["action_sequence"] == [2, 8, 10]  # type: ignore[index]
    assert evaluation["normalized_column_set"] == [2, 8, 10]  # type: ignore[index]
    assert evaluation["total_cost_musd_per_year"] == pytest.approx(3.308330)  # type: ignore[index]
    assert evaluation["total_reward"] == pytest.approx(-3.308330)  # type: ignore[index]


def test_payload_recovers_initial_q_values(experiment_payload) -> None:
    """Verify the serialized initial action values recover all three flowsheet costs.

    Inputs:
        experiment_payload: Accepted deterministic experiment document.
    Returns:
        None. The test passes when initial Q values match the required targets.
    """

    records = experiment_payload["q_values"]  # type: ignore[index]
    initial_values = {
        record["action_id"]: record["value"]
        for record in records
        if record["state"] == list(INITIAL_STATE)
    }
    assert initial_values == pytest.approx(
        {1: -3.927360, 2: -3.308330, 3: -4.102530},
        abs=1e-6,
    )


def test_experiment_payload_is_exactly_reproducible(experiment_payload) -> None:
    """Verify identical configuration produces an identical complete payload.

    Inputs:
        experiment_payload: First accepted experiment document.
    Returns:
        None. The test passes when a second run is exactly equal.
    """

    repeated = build_experiment_payload(QLearningConfig(seed=42), episodes=10_000)
    assert repeated == experiment_payload


def test_main_prints_parseable_deterministic_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify the public CLI returns success and emits only one JSON document.

    Inputs:
        capsys: Pytest standard-output capture fixture.
    Returns:
        None. The test passes when CLI output parses and contains the optimum.
    """

    status = main(["--episodes", "10000", "--seed", "42"])
    first_output = capsys.readouterr().out
    assert status == 0
    assert json.loads(first_output)["evaluation"]["action_sequence"] == [2, 8, 10]

    main(["--episodes", "10000", "--seed", "42"])
    second_output = capsys.readouterr().out
    assert second_output == first_output
