"""
Last modified time: 2026-09-23
Last modified content: Test policy-network dimensions and hard action masking
Last modified by: OpenAI Codex
File design: Focused policy and distribution unit tests
File purpose: Verify logits, legal probabilities, sampling, and mask gradients
File creator: OpenAI Codex
"""

from __future__ import annotations

import pytest
import torch

from distillation_reinforce import INITIAL_STATE, PolicyNetwork, action_mask, masked_distribution
from distillation_reinforce.environment import encode_state


def test_policy_network_produces_ten_logits_and_gradients() -> None:
    """Check that the six-input network emits trainable ten-action logits.

    Inputs:
        Initial chemical state and a two-state batch.
    Returns:
        None. Output shapes and parameter gradients are asserted.
    """

    network = PolicyNetwork()
    single_state = torch.tensor(INITIAL_STATE, dtype=torch.float32)
    batch = torch.stack((single_state, single_state))

    assert network(single_state).shape == (10,)
    assert network(batch).shape == (2, 10)

    network(single_state).sum().backward()
    assert all(parameter.grad is not None for parameter in network.parameters())


@pytest.mark.parametrize("shape", [(5,), (2, 5), (1, 2, 6)])
def test_policy_network_rejects_invalid_state_shapes(shape: tuple[int, ...]) -> None:
    """Check that policy inputs always have one six-feature state axis.

    Inputs:
        shape: Invalid state tensor shape.
    Returns:
        None. Invalid shapes raise ValueError.
    """

    with pytest.raises(ValueError, match="six features"):
        PolicyNetwork()(torch.zeros(shape))


def test_masked_distribution_assigns_probability_only_to_legal_towers() -> None:
    """Check normalization and exact zeros for an AB plus CD state.

    Inputs:
        Ten logits and the legal mask for active AB and CD streams.
    Returns:
        None. Only towers eight and ten receive positive probability.
    """

    logits = torch.arange(10, dtype=torch.float32)
    mask = action_mask(encode_state(("AB", "CD")))
    probabilities = masked_distribution(logits, mask).probs

    assert probabilities.sum().item() == pytest.approx(1.0)
    assert probabilities[7].item() > 0.0
    assert probabilities[9].item() > 0.0
    assert all(probabilities[index].item() == 0.0 for index in range(10) if not mask[index])


def test_masked_distribution_samples_only_legal_action_indices() -> None:
    """Check that categorical samples never select a masked tower.

    Inputs:
        Equal logits with only towers eight and ten legal.
    Returns:
        None. Every sampled index maps back to a feasible tower.
    """

    torch.manual_seed(17)
    mask = action_mask(encode_state(("AB", "CD")))
    distribution = masked_distribution(torch.zeros(10), mask)
    action_ids = {int(distribution.sample().item()) + 1 for _ in range(100)}
    assert action_ids == {8, 10}


def test_masked_distribution_blocks_invalid_logit_gradients() -> None:
    """Check that illegal tower logits cannot influence the policy gradient.

    Inputs:
        Trainable logits and the initial state's legal-action mask.
    Returns:
        None. Invalid tower gradients are exactly zero.
    """

    logits = torch.zeros(10, requires_grad=True)
    mask = action_mask(INITIAL_STATE)
    distribution = masked_distribution(logits, mask)
    distribution.log_prob(torch.tensor(1)).backward()

    assert logits.grad is not None
    assert logits.grad[1].item() > 0.0
    assert all(logits.grad[index].item() == 0.0 for index in range(3, 10))


@pytest.mark.parametrize(
    ("logits", "mask", "message"),
    [
        (torch.zeros(9), (True,) * 10, "ten actions"),
        (torch.zeros(10), (True,) * 9, "ten Boolean"),
        (torch.zeros(10), (False,) * 10, "terminal state"),
        (torch.tensor([float("nan")] + [0.0] * 9), (True,) * 10, "finite"),
    ],
)
def test_masked_distribution_rejects_invalid_inputs(
    logits: torch.Tensor, mask: tuple[bool, ...], message: str
) -> None:
    """Check rejection of malformed logits, masks, and terminal states.

    Inputs:
        logits: Candidate action logits.
        mask: Candidate legal-action mask.
        message: Expected validation error text.
    Returns:
        None. Invalid inputs raise ValueError.
    """

    with pytest.raises(ValueError, match=message):
        masked_distribution(logits, mask)  # type: ignore[arg-type]
