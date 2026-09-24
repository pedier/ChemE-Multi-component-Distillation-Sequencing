"""
Last modified time: 2026-09-24
Last modified content: Consolidate verified chemistry and remove redundant work
Last modified by: OpenAI Codex
File design: Policy logits and legal-action distribution
File purpose: Produce differentiable probabilities for feasible textbook towers
File creator: OpenAI Codex
"""

from __future__ import annotations

from distillation_sequencing_env.data import ACTION_COUNT, STATE_SIZE

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from distillation_reinforce.models import ActionMask


class PolicyNetwork(nn.Module):
    """Map a six-bit separation state to ten candidate-tower logits.

    Inputs:
        state: A single six-feature tensor or batch of such tensors.
    Returns:
        Ten unmasked logits per state.
    """

    def __init__(self) -> None:
        """Construct the CPU-friendly 6-to-32-to-10 ReLU policy.

        Inputs:
            None.
        Returns:
            None. Initializes trainable linear layers.
        """

        super().__init__()
        self.layers = nn.Sequential(nn.Linear(STATE_SIZE, 32), nn.ReLU(), nn.Linear(32, ACTION_COUNT))

    def forward(self, state: Tensor) -> Tensor:
        """Compute unmasked tower logits from a state tensor.

        Inputs:
            state: Float-compatible tensor with a final dimension of six.
        Returns:
            Tensor with a final dimension of ten tower logits.
        """

        if state.ndim not in (1, 2) or state.shape[-1] != STATE_SIZE:
            raise ValueError("Policy states must have six features.")

        return self.layers(state.to(dtype=torch.float32))


def masked_distribution(logits: Tensor, mask: ActionMask) -> Categorical:
    """Create a categorical policy over only feasible tower IDs.

    Inputs:
        logits: One-dimensional tensor of ten unmasked tower logits.
        mask: Boolean legal-action mask indexed by action ID minus one.
    Returns:
        Categorical distribution with zero probability on invalid towers.
    """

    # Validate the complete action interface before applying the hard mask.
    if logits.ndim != 1 or logits.numel() != ACTION_COUNT:
        raise ValueError("Policy logits must contain exactly ten actions.")

    if len(mask) != ACTION_COUNT or any(not isinstance(value, bool) for value in mask):
        raise ValueError("The action mask must contain ten Boolean values.")

    if not any(mask):
        raise ValueError("A terminal state has no action distribution.")

    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Policy logits must be finite.")

    legal = torch.tensor(mask, dtype=torch.bool, device=logits.device)
    return Categorical(logits=logits.masked_fill(~legal, -torch.inf))

