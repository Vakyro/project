"""
Pooling methods for protein sequence embeddings.

Provides different strategies to aggregate token-level representations
into a single sequence-level embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """
    Attention-based pooling over valid tokens (excluding padding).

    Learns to weight each token by importance using a gating network.
    This is particularly useful for proteins where certain regions
    (active sites, binding pockets) may be more important than others.
    """

    def __init__(self, dim, hidden=256):
        """
        Args:
            dim: Dimension of input token embeddings
            hidden: Hidden dimension for the attention gate network
        """
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, mask):
        """
        Args:
            x: Token embeddings (B, L, D)
            mask: Attention mask (B, L) where 1=valid token, 0=padding

        Returns:
            Pooled embedding (B, D)
        """
        # Compute attention scores
        scores = self.gate(x).squeeze(-1)  # (B, L)

        # Mask out padding tokens
        scores = scores.masked_fill(~mask.bool(), -1e9)

        # Normalize with softmax
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)

        # Weighted sum
        return (x * weights).sum(dim=1)  # (B, D)


def mean_pool(x, mask):
    """
    Mean pooling over valid tokens (excluding padding).

    Simple but effective baseline - averages all non-padding token embeddings.

    Args:
        x: Token embeddings (B, L, D)
        mask: Attention mask (B, L) where 1=valid token, 0=padding

    Returns:
        Pooled embedding (B, D)
    """
    # Count valid tokens per sequence
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)

    # Sum valid tokens and divide by count
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B, D)


def cls_pool(x, mask=None):
    """
    Use the CLS token embedding (first token in ESM2).

    ESM2 places a special <cls> token at position 0 which is designed
    to capture sequence-level information through self-attention.

    Args:
        x: Token embeddings (B, L, D)
        mask: Attention mask (not used, for API consistency)

    Returns:
        CLS token embedding (B, D)
    """
    return x[:, 0, :]  # First token is <cls>
