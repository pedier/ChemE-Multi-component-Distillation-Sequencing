"""
Last modified time: 2026-09-21-02:21
Last modified content: Verify masked online selection and target-network evaluation
Last modified by: OpenAI Codex
File design: Bellman target unit tests with controlled action values
File purpose: Detect illegal next actions, network-role swaps, and terminal bootstrap
File creator: OpenAI Codex
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from distillation_dqn import DistillationSequenceEnvironment, Transition, double_dqn_targets


class FixedQNetwork(nn.Module):
    """Return one controlled ten-action value vector for every state row.

    Inputs:
        values: Ten fixed Q values indexed by textbook action ID minus one.
    Returns:
        Differentiable network stub for role-separation tests.
    """

    def __init__(self, values: list[float]) -> None:
        """Store the controlled Q values as a trainable parameter.

        Inputs:
            values: Ten action values.
        Returns:
            None.
        """

        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Repeat the controlled vector for each requested state.

        Inputs:
            states: Batch of six-entry state rows.
        Returns:
            Batch of identical ten-action Q vectors.
        """

        return self.values.unsqueeze(0).expand(states.shape[0], -1)


def test_double_targets_mask_online_choice_and_use_target_evaluation() -> None:
    """Verify that online argmax is legal and target values the selected tower.

    Inputs:
        One real nonterminal and one real terminal transition.
    Returns:
        None. The test passes when both Bellman targets are correct and detached.
    """

    environment = DistillationSequenceEnvironment()
    state, _ = environment.reset()
    first = environment.step(2)
    nonterminal = Transition(
        state, 2, first.reward, first.state, first.action_mask, first.terminated
    )
    environment.step(8)
    last_state = environment.step(10)
    terminal = Transition((0, 0, 0, 1, 0, 0), 10, last_state.reward,
                          last_state.state, last_state.action_mask, last_state.terminated)

    # Illegal action 1 must lose despite its high online value; action 10 beats 8.
    online = FixedQNetwork([99.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0])
    target = FixedQNetwork([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, -5.0])
    values = double_dqn_targets(online, target, (nonterminal, terminal))

    assert values.shape == (2,)
    assert values[0].item() == pytest.approx(first.reward - 5.0)
    assert values[1].item() == pytest.approx(last_state.reward)
    assert not values.requires_grad


def test_double_targets_reject_empty_batch() -> None:
    """Require at least one transition for a Bellman target batch.

    Inputs:
        Two controlled networks and an empty transition tuple.
    Returns:
        None. The test passes when target computation raises ValueError.
    """

    network = FixedQNetwork([0.0] * 10)
    with pytest.raises(ValueError, match="nonempty"):
        double_dqn_targets(network, network, ())
