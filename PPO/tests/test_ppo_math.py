"""
Last modified time: 2026-09-24
Last modified content: Unit-test PPO configuration, GAE recursion, and clipping behavior
Last modified by: OpenAI Codex
File design: Algorithm-math unit tests
File purpose: Keep the undiscounted advantage and clipped actor objective correct
File creator: OpenAI Codex
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from distillation_ppo.ppo import PPOConfig, clipped_policy_loss, gae_advantages


def test_ppo_config_defaults_match_reviewed_plan() -> None:
    """Verify the documented default optimizer and GAE settings.

    Inputs:
        Default PPOConfig instance.
    Returns:
        None. All reviewed defaults match the implementation.
    """

    config = PPOConfig()
    assert config.learning_rate == pytest.approx(3e-4)
    assert (config.batch_episodes, config.update_epochs, config.minibatch_size) == (32, 4, 48)
    assert (config.clip_epsilon, config.gae_lambda) == pytest.approx((0.2, 0.95))
    assert (config.value_coefficient, config.entropy_coefficient) == pytest.approx((0.5, 0.01))
    assert (config.max_gradient_norm, config.seed) == (0.5, 42)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 0), ("learning_rate", float("nan")),
        ("clip_epsilon", 1), ("clip_epsilon", -0.1),
        ("gae_lambda", 1.1), ("gae_lambda", -0.1),
        ("value_coefficient", -0.1), ("entropy_coefficient", float("inf")),
        ("max_gradient_norm", 0), ("batch_episodes", 0),
        ("update_epochs", True), ("minibatch_size", 0.5), ("seed", True),
    ],
)
def test_ppo_config_rejects_invalid_settings(field: str, value: object) -> None:
    """Verify each invalid optimization setting fails at construction.

    Inputs:
        Field name and invalid field value.
    Returns:
        None. Configuration construction raises ValueError.
    """

    with pytest.raises(ValueError):
        replace(PPOConfig(), **{field: value})


def test_gae_advantages_respect_terminal_boundary_and_unit_discount() -> None:
    """Verify GAE recursion and value targets for a known three-step example.

    Inputs:
        Three rewards, three critic values, and a final terminal flag.
    Returns:
        None. Advantages and targets match hand calculations.
    """

    rewards = (-1.0, -2.0, -3.0)
    values = (-2.0, -2.0, -2.0)
    done = (False, False, True)
    advantages, returns = gae_advantages(rewards, values, done, 0.95)
    assert advantages == pytest.approx((-3.8025, -2.95, -1.0))
    assert returns == pytest.approx((-5.8025, -4.95, -3.0))

    _, monte_carlo_targets = gae_advantages(rewards, values, done, 1.0)
    assert monte_carlo_targets == pytest.approx((-6.0, -5.0, -3.0))


@pytest.mark.parametrize(
    ("rewards", "values", "done", "trace"),
    [
        ((), (), (), 0.95),
        ((-1.0,), (), (True,), 0.95),
        ((-1.0,), (0.0,), (False,), 0.95),
        ((-1.0, -2.0), (0.0, 0.0), (True, True), 0.95),
        ((float("inf"),), (0.0,), (True,), 0.95),
        ((-1.0,), (float("nan"),), (True,), 0.95),
        ((-1.0,), (0.0,), (True,), 1.1),
    ],
)
def test_gae_advantages_reject_incomplete_or_nonfinite_data(
    rewards: tuple[float, ...], values: tuple[float, ...],
    done: tuple[bool, ...], trace: float,
) -> None:
    """Verify GAE never bootstraps through an incomplete terminal episode.

    Inputs:
        Malformed reward, value, terminal, or trace data.
    Returns:
        None. GAE raises ValueError before calculating targets.
    """

    with pytest.raises(ValueError):
        gae_advantages(rewards, values, done, trace)


def test_clipped_policy_loss_handles_positive_and_negative_advantages() -> None:
    """Verify both sides of clipping and their gradients against hand values.

    Inputs:
        Ratios above and below the clipping interval with signed advantages.
    Returns:
        None. Loss and gradients reflect the PPO minimum surrogate.
    """

    new_log = torch.log(torch.tensor([1.4, 1.4, 0.6, 0.6])).requires_grad_()
    old_log = torch.zeros(4)
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])
    loss = clipped_policy_loss(new_log, old_log, advantages, 0.2)
    assert loss.item() == pytest.approx(0.1)
    loss.backward()
    assert new_log.grad is not None
    assert new_log.grad[0].item() == pytest.approx(0.0)
    assert new_log.grad[1].item() != 0.0
    assert new_log.grad[3].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("new_log", "old_log", "advantages", "epsilon"),
    [
        (torch.zeros(0), torch.zeros(0), torch.zeros(0), 0.2),
        (torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1), 0.2),
        (torch.zeros(2), torch.zeros(1), torch.zeros(2), 0.2),
        (torch.zeros(1), torch.zeros(1), torch.zeros(1), 1.0),
    ],
)
def test_clipped_policy_loss_rejects_invalid_shapes_or_radius(
    new_log: torch.Tensor, old_log: torch.Tensor,
    advantages: torch.Tensor, epsilon: float,
) -> None:
    """Verify the actor loss refuses empty, mismatched, or invalid inputs.

    Inputs:
        Parameterized invalid probability, advantage, and radius values.
    Returns:
        None. PPO loss raises ValueError.
    """

    with pytest.raises(ValueError):
        clipped_policy_loss(new_log, old_log, advantages, epsilon)
