"""
Last modified time: 2026-09-24
Last modified content: Verify actor, critic, and physically masked tower distributions
Last modified by: OpenAI Codex
File design: Neural-network and action-mask unit tests
File purpose: Check shapes, gradients, illegal-action probability, and validation
File creator: OpenAI Codex
"""

from __future__ import annotations

import pytest
import torch

from distillation_ppo.policy import ActorNetwork, CriticNetwork, masked_distribution


def test_actor_forward_handles_single_and_batched_states() -> None:
    """Verify ten differentiable tower logits per six-bit state.

    Inputs:
        Single and batched six-feature tensors.
    Returns:
        None. Output shapes and actor gradients must match expectations.
    """

    actor = ActorNetwork()
    single = actor(torch.zeros(6))
    batch = actor(torch.zeros(2, 6))
    assert single.shape == (10,)
    assert batch.shape == (2, 10)
    batch.sum().backward()
    assert any(parameter.grad is not None for parameter in actor.parameters())


def test_actor_rejects_invalid_state_shape() -> None:
    """Verify the actor refuses tensors without exactly six state features.

    Inputs:
        Invalid five-feature and three-dimensional tensors.
    Returns:
        None. Both calls raise ValueError.
    """

    actor = ActorNetwork()
    with pytest.raises(ValueError):
        actor(torch.zeros(5))
    with pytest.raises(ValueError):
        actor(torch.zeros(1, 1, 6))


def test_critic_forward_handles_single_and_batched_states() -> None:
    """Verify scalar and batched value estimates retain gradients.

    Inputs:
        Single and batched six-feature tensors.
    Returns:
        None. Output shapes and critic gradients must match expectations.
    """

    critic = CriticNetwork()
    single = critic(torch.zeros(6))
    batch = critic(torch.zeros(2, 6))
    assert single.shape == ()
    assert batch.shape == (2,)
    batch.sum().backward()
    assert any(parameter.grad is not None for parameter in critic.parameters())


def test_critic_rejects_invalid_state_shape() -> None:
    """Verify the critic refuses malformed chemical states.

    Inputs:
        Five-feature and three-dimensional tensors.
    Returns:
        None. Both calls raise ValueError.
    """

    critic = CriticNetwork()
    with pytest.raises(ValueError):
        critic(torch.zeros(5))
    with pytest.raises(ValueError):
        critic(torch.zeros(1, 1, 6))


def test_masked_distribution_zeroes_illegal_towers_in_one_or_many_states() -> None:
    """Verify legal probabilities sum to one and illegal probabilities vanish.

    Inputs:
        Uniform logits and the initial-state and AB-plus-CD masks.
    Returns:
        None. Single and batched categorical distributions match the masks.
    """

    first_mask = (True, True, True, False, False, False, False, False, False, False)
    second_mask = (False, False, False, False, False, False, False, True, False, True)
    single = masked_distribution(torch.zeros(10), first_mask)
    batch = masked_distribution(torch.zeros(2, 10), torch.tensor([first_mask, second_mask]))
    assert single.probs[:3].tolist() == pytest.approx([1 / 3] * 3)
    assert single.probs[3:].tolist() == [0.0] * 7
    assert batch.probs[0, 3:].tolist() == [0.0] * 7
    assert batch.probs[1, 7].item() == pytest.approx(0.5)
    assert batch.probs[1, 9].item() == pytest.approx(0.5)
    assert batch.probs.sum(dim=-1).tolist() == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize(
    ("logits", "mask"),
    [
        (torch.zeros(9), (True,) * 9),
        (torch.zeros(10), (False,) * 10),
        (torch.zeros(10), (1,) * 10),
        (torch.zeros(2, 10), torch.ones(2, 10)),
        (torch.zeros(2, 10), torch.ones(10, dtype=torch.bool)),
        (torch.full((10,), float("inf")), (True,) * 10),
    ],
)
def test_masked_distribution_rejects_invalid_inputs(
    logits: torch.Tensor, mask: tuple[bool, ...] | torch.Tensor,
) -> None:
    """Verify invalid shapes, types, terminal masks, and logits are rejected.

    Inputs:
        Parameterized malformed logits and masks.
    Returns:
        None. Each pair raises ValueError.
    """

    with pytest.raises(ValueError):
        masked_distribution(logits, mask)  # type: ignore[arg-type]
