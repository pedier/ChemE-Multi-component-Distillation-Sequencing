"""
Last modified time: 2026-09-24
Last modified content: Consolidate verified chemistry and remove redundant work
Last modified by: OpenAI Codex
File design: PyTorch feed-forward value and advantage streams
File purpose: Estimate ten textbook tower action values from six state bits
File creator: OpenAI Codex
"""

from __future__ import annotations

from distillation_sequencing_env.data import ACTION_COUNT, STATE_SIZE

import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    """Estimate ten action values from the fixed six-entry state.

    Inputs:
        State batches with shape ``(batch_size, 6)``.
    Returns:
        Action-value batches with shape ``(batch_size, 10)``.
    """

    def __init__(self) -> None:
        """Construct the 6-64-64 trunk and separate dueling heads.

        Inputs:
            None.
        Returns:
            None. Initializes trainable PyTorch layers.
        """

        super().__init__()

        # Shared features feed separate value and relative-advantage heads.
        self.trunk = nn.Sequential(
            nn.Linear(STATE_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(64, 1)
        self.advantage_head = nn.Linear(64, ACTION_COUNT)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute Q values using the dueling value-advantage identity.

        Inputs:
            states: Float tensor with one six-entry state per row.
        Returns:
            Float tensor with ten tower Q values per row.
        """

        if states.ndim != 2 or states.shape[1] != STATE_SIZE:
            raise ValueError("states must have shape (batch_size, 6).")

        features = self.trunk(states)
        values = self.value_head(features)
        advantages = self.advantage_head(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)

