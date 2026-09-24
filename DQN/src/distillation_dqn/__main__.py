"""
Last modified time: 2026-09-21-03:05
Last modified content: Add the package module command-line entry point
Last modified by: OpenAI Codex
File design: Thin executable package adapter
File purpose: Support python -m distillation_dqn
File creator: OpenAI Codex
"""

from distillation_dqn.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
