"""
Last modified time: 2026-09-24
Last modified content: Verify PPO command-line parsing and deterministic JSON results
Last modified by: OpenAI Codex
File design: Public command-line interface tests
File purpose: Check reproducible runs and engineering report shape
File creator: OpenAI Codex
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from distillation_ppo.cli import main, parse_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parse_args_uses_reviewed_defaults_and_overrides() -> None:
    """Verify the command-line defaults and explicit experiment settings.

    Inputs:
        Empty and overridden command-line argument sequences.
    Returns:
        None. Parsed values match the planned seed and episode budget.
    """

    defaults = parse_args([])
    options = parse_args(["--episodes", "17", "--seed", "7"])
    assert (defaults.episodes, defaults.seed) == (20_000, 42)
    assert (options.episodes, options.seed) == (17, 7)


@pytest.mark.parametrize("arguments", [
    ["--episodes", "0"], ["--episodes", "-1"],
    ["--episodes", "one"], ["--seed", "one"],
])
def test_parse_args_rejects_invalid_numbers(arguments: list[str]) -> None:
    """Verify malformed experiment settings fail during argument parsing.

    Inputs:
        Invalid episode or seed argument sequence.
    Returns:
        None. Argument parsing exits with an error.
    """

    with pytest.raises(SystemExit):
        parse_args(arguments)


def test_main_prints_json_with_true_economics(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify the CLI separates PPO diagnostics from pure-policy cost.

    Inputs:
        Three-episode training run and captured standard output.
    Returns:
        None. JSON contains settings, seven policy rows, and exact reward-cost equality.
    """

    assert main(["--episodes", "3", "--seed", "7"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["algorithm"] == "PPO"
    assert report["config"]["gamma"] == 1.0
    assert report["config"]["episodes"] == 3
    assert report["training"]["updates"] == 1
    assert len(report["policy"]) == 7
    assert report["evaluation"]["total_reward"] == pytest.approx(
        -report["evaluation"]["total_cost_musd_per_year"]
    )


def test_module_entry_point_runs_from_the_standalone_source_tree() -> None:
    """Verify python -m invokes the PPO JSON command-line entry point.

    Inputs:
        One-episode subprocess run with the local source on PYTHONPATH.
    Returns:
        None. The process exits successfully and reports a complete flowsheet.
    """

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "distillation_ppo", "--episodes", "1", "--seed", "7"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(completed.stdout)
    assert report["algorithm"] == "PPO"
    assert report["training"]["episodes"] == 1
    assert len(report["evaluation"]["action_sequence"]) == 3
