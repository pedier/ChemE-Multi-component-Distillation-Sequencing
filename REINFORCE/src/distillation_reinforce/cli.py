"""
Last modified time: 2026-09-23
Last modified content: Add the deterministic REINFORCE experiment command line
Last modified by: OpenAI Codex
File design: Argument parsing and JSON experiment output
File purpose: Run seeded training and report pure-policy chemical economics
File creator: OpenAI Codex
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Sequence

from distillation_reinforce.reinforce import ReinforceConfig
from distillation_reinforce.training import (
    DEFAULT_EPISODES,
    evaluate,
    policy_snapshot,
    train,
    training_summary,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse episode and seed settings for one reproducible experiment.

    Inputs:
        argv: Optional command line argument sequence.
    Returns:
        Namespace with positive episodes and an integer seed.
    """

    parser = argparse.ArgumentParser(description="Train masked REINFORCE on Example 17.3.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args(argv)

    if options.episodes <= 0:
        parser.error("--episodes must be a positive integer")

    return options


def main(argv: Sequence[str] | None = None) -> int:
    """Train, evaluate, and print one deterministic JSON experiment record.

    Inputs:
        argv: Optional command line argument sequence.
    Returns:
        Zero after writing the complete experiment record to standard output.
    """

    options = parse_args(argv)
    config = ReinforceConfig(seed=options.seed)
    agent = train(config=config, episodes=options.episodes)
    result = evaluate(agent)

    # Keep the original economic result separate from training diagnostics.
    report = {
        "algorithm": "REINFORCE",
        "config": {**asdict(config), "episodes": options.episodes, "gamma": 1.0},
        "training": training_summary(agent),
        "policy": policy_snapshot(agent),
        "evaluation": asdict(result),
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0
