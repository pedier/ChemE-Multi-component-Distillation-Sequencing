"""
Last modified time: 2026-09-21-02:10
Last modified content: Implement masked Double DQN bootstrap targets
Last modified by: OpenAI Codex
File design: Stateless Bellman target calculation
File purpose: Separate online action selection from target-network evaluation
File creator: OpenAI Codex
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from distillation_dqn.replay import Transition


def double_dqn_targets(
    online_network: nn.Module,
    target_network: nn.Module,
    transitions: Sequence[Transition],
) -> torch.Tensor:
    """Compute undiscounted masked Double DQN targets on CPU.

    Inputs:
        online_network: Network that selects the next feasible tower.
        target_network: Frozen network that evaluates that tower.
        transitions: Nonempty batch of valid observed transitions.
    Returns:
        Detached float tensor with one Bellman target per transition.
    """

    if not transitions:
        raise ValueError("transitions must be nonempty.")

    # Terminal rows may have no legal tower; their gathered value is discarded.
    next_states = torch.tensor([item.next_state for item in transitions], dtype=torch.float32)
    next_masks = torch.tensor([item.next_action_mask for item in transitions], dtype=torch.bool)
    rewards = torch.tensor([item.reward for item in transitions], dtype=torch.float32)
    terminated = torch.tensor([item.terminated for item in transitions], dtype=torch.bool)

    with torch.no_grad():
        online_values = online_network(next_states).masked_fill(~next_masks, -torch.inf)
        next_action_indices = online_values.argmax(dim=1, keepdim=True)
        target_values = target_network(next_states).gather(1, next_action_indices).squeeze(1)
        future_values = torch.where(terminated, 0.0, target_values)

    return rewards + future_values
