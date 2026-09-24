"""
Last modified time: 2026-09-24
Last modified content: Consolidate verified chemistry and remove redundant work
Last modified by: OpenAI Codex
File design: Small CPU-friendly neural networks and legal-action distribution
File purpose: Produce tower probabilities and remaining-cost value estimates
File creator: OpenAI Codex
"""

from __future__ import annotations

from distillation_sequencing_env.data import ACTION_COUNT, STATE_SIZE

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from distillation_ppo.models import ActionMask


class ActorNetwork(nn.Module):
    """Map chemical states to unmasked textbook-tower logits.

    Inputs:
        state: One six-feature state or a batch of six-feature states.
    Returns:
        Ten logits for each input state.
    """

    def __init__(self) -> None:
        """Construct the 6-to-32-to-10 actor.

        Inputs:
            None.
        Returns:
            None. Initializes the actor layers.
        """

        super().__init__()
        self.layers = nn.Sequential(nn.Linear(STATE_SIZE, 32), nn.ReLU(), nn.Linear(32, ACTION_COUNT))

    def forward(self, state: Tensor) -> Tensor:
        """Compute logits for one or more chemical states.

        Inputs:
            state: Float-compatible tensor with six features per state.
        Returns:
            Tensor with ten unmasked logits per state.
        """

        if state.ndim not in (1, 2) or state.shape[-1] != STATE_SIZE:
            raise ValueError("Actor states must have six features.")

        return self.layers(state.to(dtype=torch.float32))


class CriticNetwork(nn.Module):
    """Estimate the undiscounted remaining reward of a chemical state.

    Inputs:
        state: One six-feature state or a batch of six-feature states.
    Returns:
        One scalar state value per input state.
    """

    def __init__(self) -> None:
        """Construct the 6-to-32-to-1 critic.

        Inputs:
            None.
        Returns:
            None. Initializes the critic layers.
        """

        super().__init__()
        self.layers = nn.Sequential(nn.Linear(STATE_SIZE, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, state: Tensor) -> Tensor:
        """Compute a scalar value for each input state.

        Inputs:
            state: Float-compatible tensor with six features per state.
        Returns:
            Scalar for a single state or one-dimensional batch of values.
        """

        if state.ndim not in (1, 2) or state.shape[-1] != STATE_SIZE:
            raise ValueError("Critic states must have six features.")

        return self.layers(state.to(dtype=torch.float32)).squeeze(-1)


def masked_distribution(logits: Tensor, mask: ActionMask | Tensor) -> Categorical:
    """Build a categorical distribution over only physically feasible towers.

    Inputs:
        logits: Ten logits for one state or a batch of states.
        mask: Boolean mask with the same shape as logits.
    Returns:
        Categorical distribution with zero probability for illegal towers.
    """

    # Validate complete action dimensions and feasibility before hard masking.
    if logits.ndim not in (1, 2) or logits.shape[-1] != ACTION_COUNT:
        raise ValueError("Policy logits must have ten actions per state.")

    if isinstance(mask, Tensor) and mask.dtype != torch.bool:
        raise ValueError("Action masks must contain Boolean values.")

    if not isinstance(mask, Tensor) and any(not isinstance(value, bool) for value in mask):
        raise ValueError("Action masks must contain Boolean values.")

    legal = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if legal.shape != logits.shape or not bool(legal.any(dim=-1).all()):
        raise ValueError("Each nonterminal state needs a matching nonempty mask.")

    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Policy logits must be finite.")

    return Categorical(logits=logits.masked_fill(~legal, -torch.inf))
