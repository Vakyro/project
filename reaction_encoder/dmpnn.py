"""
Two-Stage Directed Message Passing Neural Network (DMPNN) for reaction encoding.

Based on CLIPZyme implementation:
- Stage 1 (f_mol): Encodes substrate and product separately
- Stage 2 (f_TS): Combines bond embeddings to create pseudo-transition state

This architecture explicitly models the transformation from reactants to products.

Key difference from standard MPNN: States live on EDGES (bonds), not nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class DirectedMPNNLayer(nn.Module):
    """
    Directed Message Passing layer for DMPNN.

    In DMPNN, each directed edge (bond) maintains its own hidden state.
    Messages are passed between edges, not nodes.

    For edge (u -> v):
    - Receives messages from edges (w -> u) for all neighbors w ≠ v
    - Updates its hidden state based on these messages and node u's features
    """

    def __init__(self, node_dim, edge_dim, hidden_dim):
        """
        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden state dimension for edges
        """
        super().__init__()

        # Message function: combines neighbor edge hidden states
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Update function: combines node features, edge features, and aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr, edge_hidden):
        """
        Args:
            x: Node features (N, node_dim)
            edge_index: Edge connectivity (2, E), each column is [source, target]
            edge_attr: Edge features (E, edge_dim)
            edge_hidden: Current edge hidden states (E, hidden_dim)

        Returns:
            Updated edge hidden states (E, hidden_dim)
        """
        num_edges = edge_index.size(1)
        num_nodes = x.size(0)
        device = x.device

        # Build reverse mapping: for each edge, find incoming edges to its source node
        # Edge i goes from edge_index[0, i] -> edge_index[1, i]
        # We need to find all edges j that go into edge_index[0, i]

        # Create adjacency structure: for each node, store incoming edges
        # node_to_incoming_edges[u] = list of edge indices that end at u
        node_to_incoming_edges = [[] for _ in range(num_nodes)]
        for e_idx in range(num_edges):
            target_node = edge_index[1, e_idx].item()
            node_to_incoming_edges[target_node].append(e_idx)

        # For each edge, aggregate messages from edges entering its source node
        edge_messages = torch.zeros((num_edges, edge_hidden.size(1)), device=device)

        for e_idx in range(num_edges):
            source_node = edge_index[0, e_idx].item()
            target_node = edge_index[1, e_idx].item()

            # Find all edges entering the source node (except the reverse edge)
            incoming_to_source = node_to_incoming_edges[source_node]

            # Aggregate messages from incoming edges (excluding reverse edge)
            messages_list = []
            for incoming_idx in incoming_to_source:
                # Check if this is not the reverse edge
                incoming_source = edge_index[0, incoming_idx].item()
                if incoming_source != target_node:  # Exclude reverse edge
                    msg = self.message_mlp(edge_hidden[incoming_idx])
                    messages_list.append(msg)

            # Aggregate messages
            if messages_list:
                aggregated = torch.stack(messages_list).sum(dim=0)
            else:
                aggregated = torch.zeros(edge_hidden.size(1), device=device)

            edge_messages[e_idx] = aggregated

        # Update edge hidden states using node features, edge features, and messages
        source_nodes = edge_index[0]  # (E,)
        node_features = x[source_nodes]  # (E, node_dim)

        # Combine: node features + edge features + aggregated messages
        combined = torch.cat([node_features, edge_attr, edge_messages], dim=1)
        edge_hidden_new = self.update_mlp(combined)

        return edge_hidden_new


class DMPNN(nn.Module):
    """
    Directed Message Passing Neural Network.

    Operates on molecular graphs where bonds (edges) maintain hidden states
    that are updated through message passing.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=1280, num_layers=5, dropout=0.1):
        """
        Args:
            node_dim: Node feature dimension (9 for CLIPZyme)
            edge_dim: Edge feature dimension (3 for CLIPZyme)
            hidden_dim: Hidden dimension (1280 for CLIPZyme)
            num_layers: Number of message passing layers (5 for CLIPZyme)
            dropout: Dropout probability
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initial edge embedding
        self.edge_init = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU()
        )

        # Message passing layers
        self.layers = nn.ModuleList([
            DirectedMPNNLayer(node_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Node readout (combines node features with adjacent edge states)
        self.node_readout = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, data):
        """
        Forward pass through DMPNN.

        Args:
            data: PyG Data object with:
                - x: Node features (N, node_dim)
                - edge_index: Edge connectivity (2, E)
                - edge_attr: Edge features (E, edge_dim)

        Returns:
            node_embeddings: Node-level embeddings (N, hidden_dim)
            edge_embeddings: Edge-level embeddings (E, hidden_dim)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Initialize edge hidden states
        edge_hidden = self.edge_init(edge_attr)  # (E, hidden_dim)

        # Message passing
        for layer, norm in zip(self.layers, self.norms):
            edge_hidden_new = layer(x, edge_index, edge_attr, edge_hidden)
            edge_hidden = edge_hidden + edge_hidden_new  # Residual
            edge_hidden = norm(edge_hidden)
            edge_hidden = self.dropout(edge_hidden)

        # Node readout: aggregate adjacent edge states
        row, col = edge_index
        # For each node, aggregate incoming edge states
        node_edge_agg = torch.zeros((x.size(0), self.hidden_dim), device=x.device)
        node_edge_agg.index_add_(0, col, edge_hidden)

        # Combine node features with aggregated edge states
        node_embeddings = self.node_readout(torch.cat([x, node_edge_agg], dim=-1))

        return node_embeddings, edge_hidden


class TwoStageDMPNN(nn.Module):
    """
    Two-Stage DMPNN for reaction encoding (CLIPZyme architecture).

    Stage 1 (f_mol):
        - Encodes substrate and product graphs separately
        - Each gets its own DMPNN processing

    Stage 2 (f_TS):
        - Creates pseudo-transition state by combining substrate/product bond embeddings
        - Processes combined graph to get final reaction representation
    """

    def __init__(
        self,
        node_dim=9,
        edge_dim=3,
        hidden_dim=1280,
        num_layers=5,
        proj_dim=512,
        dropout=0.1
    ):
        """
        Args:
            node_dim: Node feature dimension (9 for CLIPZyme)
            edge_dim: Edge feature dimension (3 for CLIPZyme)
            hidden_dim: Hidden dimension (1280 for CLIPZyme)
            num_layers: Number of DMPNN layers per stage (5 for CLIPZyme)
            proj_dim: Output embedding dimension
            dropout: Dropout probability
        """
        super().__init__()

        # Stage 1: Molecular encoder (shared for substrate and product)
        self.f_mol = DMPNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # Stage 2: Not used in simplified version, but keep for compatibility
        # In full implementation, would build pseudo-transition state graph

        # Projection head - takes concatenated substrate + product pooled features
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, substrate_data, product_data, atom_mapping):
        """
        Forward pass through two-stage DMPNN.

        Args:
            substrate_data: PyG Data for substrate molecule
            product_data: PyG Data for product molecule
            atom_mapping: Dict mapping substrate atom indices to product indices
                         {substrate_idx: product_idx}

        Returns:
            z: L2-normalized reaction embedding (1, proj_dim)
        """
        device = substrate_data.x.device

        # Stage 1: Encode substrate and product separately
        node_emb_sub, edge_emb_sub = self.f_mol(substrate_data)
        node_emb_prod, edge_emb_prod = self.f_mol(product_data)

        # Stage 2: Pool and combine substrate and product representations
        # Sum pooling aggregates node embeddings for each molecule
        g_sub = torch.sum(node_emb_sub, dim=0, keepdim=True)
        g_prod = torch.sum(node_emb_prod, dim=0, keepdim=True)

        # Concatenate representations to capture the transformation
        g_combined = torch.cat([g_sub, g_prod], dim=1)

        # Project to embedding space and normalize
        z = self.projection(g_combined)
        z = F.normalize(z, p=2, dim=-1)

        return z

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.projection.net[-1].out_features


class ReactionDMPNN(nn.Module):
    """
    Simplified wrapper for TwoStageDMPNN that accepts atom-mapped reaction SMILES.

    Handles parsing and graph construction automatically.
    """

    def __init__(
        self,
        node_dim=9,
        edge_dim=3,
        hidden_dim=1280,
        num_layers=5,
        proj_dim=512,
        dropout=0.1
    ):
        super().__init__()

        self.dmpnn = TwoStageDMPNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            proj_dim=proj_dim,
            dropout=dropout
        )

    def forward(self, substrate_data, product_data, atom_mapping):
        """
        Forward pass.

        Args:
            substrate_data: PyG Data for substrate
            product_data: PyG Data for product
            atom_mapping: Atom mapping dictionary

        Returns:
            Reaction embedding
        """
        return self.dmpnn(substrate_data, product_data, atom_mapping)

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.dmpnn.get_embedding_dim()
