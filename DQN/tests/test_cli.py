"""
Last modified time: 2026-09-21-03:18
Last modified content: Test DQN JSON payload, stable ordering, and module CLI
Last modified by: OpenAI Codex
File design: Serialization and command-line unit tests
File purpose: Verify reproducible public experiment output without long runs
File creator: OpenAI Codex
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

import distillation_dqn.cli as cli_module
from distillation_dqn import DQNConfig, INITIAL_STATE, TERMINAL_STATE, action_mask


def test_ordered_states_cover_all_reachable_decisions() -> None:
    """Verify breadth-first enumeration of the seven decision states.

    Inputs:
        The deterministic textbook environment graph.
    Returns:
        None. The test passes for seven unique nonterminal states.
    """

    states = cli_module._ordered_nonterminal_states()
    assert len(states) == 7
    assert states[0] == INITIAL_STATE
    assert TERMINAL_STATE not in states
    assert len(set(states)) == len(states)
    assert sum(sum(action_mask(state)) for state in states) == 12


def test_payload_contains_config_policy_q_values_and_economics() -> None:
    """Check complete JSON-compatible output after a short real training run.

    Inputs:
        Seeded DQN configuration and two online episodes.
    Returns:
        None. The test passes when records have the agreed structure.
    """

    payload = cli_module.build_experiment_payload(DQNConfig(seed=9), episodes=2)
    assert set(payload) == {"config", "policy", "q_values", "evaluation"}
    assert payload["config"]["episodes"] == 2
    assert payload["config"]["seed"] == 9
    assert payload["config"]["discount_factor"] == 1.0
    assert payload["config"]["epsilon_decay"] == 0.995
    assert payload["config"]["device"] == "cpu"
    assert len(payload["policy"]) == 7
    assert len(payload["q_values"]) == 12

    evaluation = payload["evaluation"]
    assert evaluation["final_state"] == list(TERMINAL_STATE)
    assert len(evaluation["action_sequence"]) == 3
    assert evaluation["normalized_column_set"] == sorted(evaluation["action_sequence"])
    assert evaluation["total_reward"] == pytest.approx(-evaluation["total_cost_musd_per_year"])
    json.dumps(payload, sort_keys=True)


def test_main_prints_identical_json_for_same_seed_and_episodes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify repeated command runs produce byte-identical JSON.

    Inputs:
        Two five-episode runs using the same seed.
    Returns:
        None. The test passes when parsed and raw output agree.
    """

    assert cli_module.main(["--episodes", "5", "--seed", "17"]) == 0
    first = capsys.readouterr().out
    assert cli_module.main(["--episodes", "5", "--seed", "17"]) == 0
    second = capsys.readouterr().out
    assert first == second
    assert json.loads(first)["config"]["seed"] == 17


@pytest.mark.parametrize("arguments", [["--episodes", "0"], ["--episodes", "-1"],
                                             ["--episodes", "abc"]])
def test_main_rejects_bad_episode_argument(arguments: list[str]) -> None:
    """Reject invalid episode counts before starting a training run.

    Inputs:
        arguments: CLI tokens with invalid episodes.
    Returns:
        None. The test passes when argparse exits with an error.
    """

    with pytest.raises(SystemExit) as error:
        cli_module.main(arguments)
    assert error.value.code == 2


def test_module_entry_point_outputs_json() -> None:
    """Smoke-test the installed Python module command.

    Inputs:
        One-episode subprocess using the current test interpreter.
    Returns:
        None. The test passes when module execution returns valid JSON.
    """

    result = subprocess.run(
        [sys.executable, "-m", "distillation_dqn", "--episodes", "1", "--seed", "3"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["config"]["episodes"] == 1
    assert payload["evaluation"]["final_state"] == list(TERMINAL_STATE)
