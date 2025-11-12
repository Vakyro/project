"""
Enhanced ReactionGNN model with attention pooling and projection head.

Improvements over base model:
- GlobalAttention pooling instead of mean pooling
- ProjectionHead with Dropout and LayerNorm for better separation
- Support for dual-branch architecture (full graph + change-only graph)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, GlobalAttention
from torch_geometric.data import Data, Batch


class MLP(nn.Module):
    """Multi-layer perceptron for GNN message passing."""

    def __init__(self, in_dim, hidden_dim, out_dim, layers=2):
        super().__init__()
        self.in_channels = in_dim  # For GINEConv compatibility

        if layers == 1:
            self.net = nn.Sequential(nn.Linear(in_dim, out_dim))
        elif layers == 2:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            for _ in range(layers - 2):
                mods.append(nn.Linear(hidden_dim, hidden_dim))
                mods.append(nn.ReLU())
            mods.append(nn.Linear(hidden_dim, out_dim))
            self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class ProjectionHead(nn.Module):
    """
    Projection head with Dropout and LayerNorm for better embedding separation.

    This helps prevent the "all embeddings too similar" problem by:
    - Adding regularization through dropout
    - Normalizing intermediate representations
    - Using deeper projection
    """

    def __init__(self, in_dim, proj_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, g):
        z = self.net(g)
        # L2 normalize for cosine similarity
        return F.normalize(z, p=2, dim=-1)


class ReactionGNNEnhanced(nn.Module):
    """
    Enhanced ReactionGNN with attention pooling and projection head.

    Key improvements:
    - GlobalAttention instead of mean pooling (learns which atoms are important)
    - ProjectionHead for better embedding separation
    - Supports both original (17-dim) and enhanced (28-dim) node features
    """

    def __init__(
        self,
        x_dim,
        e_dim,
        hidden=128,
        layers=3,
        out_dim=256,
        dropout=0.2,
        use_attention=True
    ):
        super().__init__()

        self.use_attention = use_attention

        # Edge feature MLP
        self.edge_mlp = MLP(e_dim, hidden, hidden)

        # GNN layers (GINE)
        self.convs = nn.ModuleList()
        for i in range(layers):
            self.convs.append(
                GINEConv(
                    nn=MLP(x_dim if i == 0 else hidden, hidden, hidden),
                    edge_dim=hidden
                )
            )

        # Layer normalization after each GNN layer
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        # Pooling
        if use_attention:
            # Attention gate learns to weight nodes by importance
            gate_nn = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1)
            )
            self.pool = GlobalAttention(gate_nn=gate_nn)
        else:
            self.pool = global_mean_pool

        # Projection head
        self.projection = ProjectionHead(hidden, out_dim, dropout)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, "batch", None)

        # Process edge features
        edge_feat = self.edge_mlp(edge_attr)

        # GNN layers
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_feat)
            h = norm(h)
            h = F.relu(h)

        # Pooling
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        if self.use_attention:
            g = self.pool(h, batch)
        else:
            g = global_mean_pool(h, batch)

        # Projection head with normalization
        z = self.projection(g)

        return z

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.projection.net[-1].out_features


class DualBranchReactionGNN(nn.Module):
    """
    Dual-branch architecture combining full graph and change-only graph.

    The model processes two graphs:
    1. Full union graph (all atoms and bonds)
    2. Change-only graph (only reactive centers)

    This forces the model to focus on both the overall structure
    and the specific transformation.
    """

    def __init__(
        self,
        x_dim,
        e_dim,
        hidden=128,
        layers=3,
        out_dim=256,
        dropout=0.2
    ):
        super().__init__()

        # Two encoders: one for full graph, one for change-only
        self.encoder_full = ReactionGNNEnhanced(
            x_dim=x_dim,
            e_dim=e_dim,
            hidden=hidden,
            layers=layers,
            out_dim=hidden,  # Output intermediate dim
            dropout=dropout,
            use_attention=True
        )

        self.encoder_change = ReactionGNNEnhanced(
            x_dim=x_dim,
            e_dim=e_dim,
            hidden=hidden,
            layers=layers,
            out_dim=hidden,  # Output intermediate dim
            dropout=dropout,
            use_attention=True
        )

        # Fusion layer to combine both branches
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, data_full, data_change):
        """
        Args:
            data_full: PyG Data with full graph
            data_change: PyG Data with change-only graph

        Returns:
            z: Normalized embedding (out_dim)
        """
        # Encode both graphs
        z_full = self.encoder_full(data_full)
        z_change = self.encoder_change(data_change)

        # Concatenate and fuse
        z_cat = torch.cat([z_full, z_change], dim=1)
        z = self.fusion(z_cat)

        # L2 normalize
        z = F.normalize(z, p=2, dim=-1)

        return z

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.fusion[-1].out_features


def create_enhanced_model(x_dim, e_dim, hidden=128, layers=3, out_dim=256, model_type='attention'):
    """
    Factory function to create enhanced models.

    Args:
        x_dim: Node feature dimension
        e_dim: Edge feature dimension
        hidden: Hidden dimension
        layers: Number of GNN layers
        out_dim: Output embedding dimension
        model_type: 'attention' or 'dual_branch'

    Returns:
        ReactionGNNEnhanced or DualBranchReactionGNN
    """
    if model_type == 'attention':
        return ReactionGNNEnhanced(
            x_dim=x_dim,
            e_dim=e_dim,
            hidden=hidden,
            layers=layers,
            out_dim=out_dim,
            dropout=0.2,
            use_attention=True
        )
    elif model_type == 'dual_branch':
        return DualBranchReactionGNN(
            x_dim=x_dim,
            e_dim=e_dim,
            hidden=hidden,
            layers=layers,
            out_dim=out_dim,
            dropout=0.2
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
