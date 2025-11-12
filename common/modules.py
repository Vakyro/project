"""
Shared neural network modules for CLIPZyme.

Contains unified implementations of:
- ProjectionHead: Maps embeddings to shared space for CLIP loss
- MLP: Multi-layer perceptron for various tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class MLP(nn.Module):
    """
    Configurable multi-layer perceptron.

    Unified implementation that works for both GNN message passing
    and general feed-forward networks.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = 'relu',
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False
    ):
        """
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension (defaults to in_dim)
            out_dim: Output dimension (defaults to in_dim)
            num_layers: Number of layers (must be >= 1)
            activation: Activation function ('relu', 'gelu', 'tanh')
            dropout: Dropout probability (0 = no dropout)
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        assert num_layers >= 1, "num_layers must be at least 1"

        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim

        # For PyG GINEConv compatibility
        self.in_channels = in_dim

        # Choose activation
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = []

        if num_layers == 1:
            # Single layer
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            # First layer
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(act_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            # Output layer
            layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., in_dim)

        Returns:
            Output tensor of shape (..., out_dim)
        """
        return self.net(x)


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.

    Maps embeddings from encoder space to a shared embedding space
    where CLIP loss is computed. Includes normalization and regularization
    for stable training.
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 512,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        activation: str = 'relu'
    ):
        """
        Args:
            in_dim: Input embedding dimension
            proj_dim: Output projection dimension
            hidden_dim: Hidden layer size (defaults to in_dim)
            dropout: Dropout probability for regularization
            use_batch_norm: Use batch normalization (not recommended with small batches)
            use_layer_norm: Use layer normalization (recommended)
            activation: Activation function ('relu', 'gelu')
        """
        super().__init__()

        hidden_dim = hidden_dim or in_dim

        # Choose activation
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build projection network
        layers = [
            nn.Linear(in_dim, hidden_dim),
            act_fn(),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))

        layers.append(nn.Linear(hidden_dim, proj_dim))

        self.net = nn.Sequential(*layers)
        self._proj_dim = proj_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with L2 normalization.

        Args:
            x: Input embedding of shape (B, in_dim)

        Returns:
            Projected and L2-normalized embedding of shape (B, proj_dim)
            with unit norm for cosine similarity.
        """
        z = self.net(x)
        # L2 normalize for cosine similarity
        return F.normalize(z, p=2, dim=-1)

    @property
    def proj_dim(self) -> int:
        """Get projection dimension."""
        return self._proj_dim


class ResidualMLP(nn.Module):
    """
    MLP with residual connections.

    Useful for deeper networks to avoid vanishing gradients.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        """
        Args:
            dim: Input/output dimension (must be same for residual)
            hidden_dim: Hidden dimension (defaults to dim)
            num_layers: Number of layers
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()

        hidden_dim = hidden_dim or dim

        self.mlp = MLP(
            in_dim=dim,
            hidden_dim=hidden_dim,
            out_dim=dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Output tensor of shape (..., dim)
        """
        return self.layer_norm(x + self.mlp(x))


class AttentionPooling(nn.Module):
    """
    Learnable attention-based pooling.

    Computes attention weights over sequence and pools features.
    """

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: Dimension of input features
        """
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Attention pooling.

        Args:
            x: Input features of shape (B, L, D)
            mask: Optional mask of shape (B, L) where 1=valid, 0=padding

        Returns:
            Pooled features of shape (B, D)
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (B, L)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get weights
        weights = F.softmax(scores, dim=1)  # (B, L)

        # Weighted sum
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, D)

        return pooled


# Export all modules
__all__ = [
    'MLP',
    'ProjectionHead',
    'ResidualMLP',
    'AttentionPooling',
]
