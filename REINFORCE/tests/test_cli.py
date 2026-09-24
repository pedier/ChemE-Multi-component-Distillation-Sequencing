"""
Last modified time: 2026-09-23
Last modified content: Test deterministic JSON output and module command line behavior
Last modified by: OpenAI Codex
File design: Command line argument and experiment reporting tests
File purpose: Verify reproducible records and executable package entry point
File creator: OpenAI Codex
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from distillation_reinforce.cli import main, parse_args


def test_parse_args_uses_default_training_settings() -> None:
    """Check that the public command line defaults match the experiment plan.

    Inputs:
        Empty argument list.
    Returns:
        None. The parsed episode budget and seed equal their defaults.
    """

    options = parse_args([])
    assert options.episodes == 20_000
    assert options.seed == 42


def test_parse_args_accepts_custom_episode_budget_and_seed() -> None:
    """Check that users can choose a short reproducible experiment.

    Inputs:
        Two explicit command line flags.
    Returns:
        None. Both values are parsed as integers.
    """

    options = parse_args(["--episodes", "4", "--seed", "7"])
    assert (options.episodes, options.seed) == (4, 7)


@pytest.mark.parametrize("value", ["0", "-2", "bad"])
def test_parse_args_rejects_invalid_episode_counts(value: str) -> None:
    """Check that invalid training budgets exit before any experiment begins.

    Inputs:
        value: Nonpositive or noninteger episode count.
    Returns:
        None. Argument parsing raises SystemExit.
    """

    with pytest.raises(SystemExit):
        parse_args(["--episodes", value])


def test_main_prints_stable_complete_json_for_seeded_short_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """Check reproducible JSON content for two identical online runs.

    Inputs:
        Two three-episode experiments using the same random seed.
    Returns:
        None. Both records match and contain economics and all policy states.
    """

    assert main(["--episodes", "3", "--seed", "7"]) == 0
    first_output = capsys.readouterr().out
    assert main(["--episodes", "3", "--seed", "7"]) == 0
    second_output = capsys.readouterr().out
    record = json.loads(first_output)

    assert first_output == second_output
    assert record["algorithm"] == "REINFORCE"
    assert record["config"]["gamma"] == 1.0
    assert record["config"]["episodes"] == 3
    assert record["training"]["updates"] == 1
    assert len(record["policy"]) == 7
    assert len(record["evaluation"]["action_sequence"]) == 3
    assert record["evaluation"]["total_reward"] == pytest.approx(
        -record["evaluation"]["total_cost_musd_per_year"]
    )


def test_python_module_entry_point_emits_json() -> None:
    """Check the installed package's module execution adapter.

    Inputs:
        Short subprocess invocation of python -m distillation_reinforce.
    Returns:
        None. Standard output contains a parseable experiment record.
    """

    completed = subprocess.run(
        [sys.executable, "-m", "distillation_reinforce", "--episodes", "2", "--seed", "9"],
        capture_output=True,
        text=True,
        check=True,
    )
    record = json.loads(completed.stdout)

    assert record["training"]["episodes"] == 2
    assert record["config"]["seed"] == 9
    assert len(record["policy"]) == 7
