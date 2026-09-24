"""
Last modified time: 2026-09-21-02:18
Last modified content: Verify dueling dimensions, identity, gradients, and input checks
Last modified by: OpenAI Codex
File design: Focused PyTorch network unit tests
File purpose: Protect the fixed six-state ten-action dueling architecture
File creator: OpenAI Codex
"""

from __future__ import annotations

import pytest
import torch

from distillation_dqn.network import DuelingQNetwork


def test_network_has_required_shapes_and_trainable_heads() -> None:
    """Verify the 6-64-64 trunk and both ten-action dueling streams.

    Inputs:
        A newly constructed network and a batch of four states.
    Returns:
        None. The test passes when dimensions and gradients are correct.
    """

    network = DuelingQNetwork()
    assert network.trunk[0].in_features == 6
    assert network.trunk[0].out_features == 64
    assert network.trunk[2].in_features == 64
    assert network.trunk[2].out_features == 64
    assert network.value_head.out_features == 1
    assert network.advantage_head.out_features == 10

    outputs = network(torch.ones((4, 6), dtype=torch.float32))
    assert outputs.shape == (4, 10)
    outputs.sum().backward()
    assert network.value_head.weight.grad is not None
    assert network.advantage_head.weight.grad is not None


def test_forward_uses_mean_centered_advantage() -> None:
    """Verify Q equals value plus advantage minus mean advantage.

    Inputs:
        A network whose value and advantage biases are controlled.
    Returns:
        None. The test passes when every action has the expected value.
    """

    network = DuelingQNetwork()
    with torch.no_grad():
        network.value_head.weight.zero_()
        network.value_head.bias.fill_(2.0)
        network.advantage_head.weight.zero_()
        network.advantage_head.bias.copy_(torch.arange(10, dtype=torch.float32))

    values = network(torch.zeros((2, 6), dtype=torch.float32))
    expected = 2.0 + torch.arange(10, dtype=torch.float32) - 4.5
    assert torch.allclose(values[0], expected)
    assert torch.allclose(values[1], expected)


@pytest.mark.parametrize("shape", [(6,), (2, 5), (1, 6, 1)])
def test_forward_rejects_wrong_state_shape(shape: tuple[int, ...]) -> None:
    """Reject a tensor that is not a batch of six-entry states.

    Inputs:
        shape: Invalid tensor dimensions.
    Returns:
        None. The test passes when forward raises ValueError.
    """

    with pytest.raises(ValueError, match="shape"):
        DuelingQNetwork()(torch.zeros(shape))
