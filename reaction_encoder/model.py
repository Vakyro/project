"""
Graph Neural Network model for encoding reaction transition states.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.in_channels = in_dim  # For GINEConv compatibility
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReactionGNN(nn.Module):
    """
    Graph Neural Network for encoding chemical reactions.

    Uses GINE (Graph Isomorphism Network with Edge features) to process
    the transition state graph and produces a fixed-size embedding.

    Args:
        x_dim: Dimension of node features
        e_dim: Dimension of edge features
        hidden: Hidden dimension for GNN layers
        layers: Number of GNN layers
        out_dim: Output embedding dimension
    """

    def __init__(self, x_dim: int, e_dim: int, hidden: int = 128, layers: int = 3, out_dim: int = 256):
        super().__init__()

        # GNN convolution layers
        self.convs = nn.ModuleList()
        for i in range(layers):
            mlp = MLP(x_dim if i == 0 else hidden, hidden, hidden)
            self.convs.append(GINEConv(nn=mlp, edge_dim=e_dim))

        # Layer normalization for each GNN layer
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        # Readout MLP to produce final embedding
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, data):
        """
        Forward pass through the GNN.

        Args:
            data: PyTorch Geometric Data object with:
                - x: node features [num_nodes, x_dim]
                - edge_index: edge connectivity [2, num_edges]
                - edge_attr: edge features [num_edges, e_dim]
                - batch: batch assignment (optional, for batched graphs)

        Returns:
            Reaction embedding tensor [batch_size, out_dim], L2-normalized
        """
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, "batch", None)

        # Apply GNN layers
        h = x
        for conv, ln in zip(self.convs, self.norms):
            h = conv(h, ei, ea)
            h = ln(torch.relu(h))

        # Global pooling
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        g = global_mean_pool(h, batch)

        # Readout to embedding
        z = self.readout(g)

        # L2 normalize for cosine similarity
        z = nn.functional.normalize(z, p=2, dim=-1)

        return z

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.readout[-1].out_features
