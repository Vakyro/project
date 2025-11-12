"""
E(n)-Equivariant Graph Neural Network (EGNN) for protein encoding.

Based on:
- "E(n) Equivariant Graph Neural Networks" (Satorras et al., 2021)
- CLIPZyme implementation details

EGNN is equivariant to translations, rotations, and reflections,
making it ideal for processing 3D protein structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import math


class SinusoidalDistanceEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings for pairwise distances.

    Similar to transformer positional encodings but for continuous distances.
    Uses theta=10,000 as in the CLIPZyme paper.
    """

    def __init__(self, num_embeddings=16, theta=10000.0):
        """
        Args:
            num_embeddings: Number of sinusoidal features (should be even)
            theta: Base for frequency calculation (default: 10,000)
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.theta = theta

        # Precompute frequencies
        freqs = torch.exp(
            -math.log(theta) * torch.arange(0, num_embeddings, 2, dtype=torch.float) / num_embeddings
        )
        self.register_buffer('freqs', freqs)

    def forward(self, distances):
        """
        Compute sinusoidal distance embeddings.

        Args:
            distances: Pairwise distances, shape (num_edges,)

        Returns:
            Sinusoidal embeddings, shape (num_edges, num_embeddings)
        """
        angles = distances.unsqueeze(-1) * self.freqs.unsqueeze(0)
        sin_embed = torch.sin(angles)
        cos_embed = torch.cos(angles)
        embeddings = torch.stack([sin_embed, cos_embed], dim=2).flatten(1)
        return embeddings


class EGNNLayer(MessagePassing):
    """
    Single layer of E(n)-Equivariant Graph Neural Network.

    Updates both node features (invariant) and coordinates (equivariant).
    Message passing preserves E(n) equivariance through careful design.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, update_coords=True):
        """
        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension (including distance embeddings)
            hidden_dim: Hidden layer dimension
            update_coords: Whether to update coordinates (set False for last layer)
        """
        super().__init__(aggr='mean')

        self.update_coords = update_coords

        # Message MLP: combines node features and edge attributes
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Coordinate update MLP (if enabled)
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1, bias=False)
            )

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, h, pos, edge_index, edge_attr):
        """
        Args:
            h: Node features (N, node_dim)
            pos: Node coordinates (N, 3)
            edge_index: Edge connectivity (2, E)
            edge_attr: Edge attributes including distance embeddings (E, edge_dim)

        Returns:
            h_new: Updated node features (N, node_dim)
            pos_new: Updated coordinates (N, 3)
        """
        # Compute pairwise differences and distances
        row, col = edge_index
        coord_diff = pos[row] - pos[col]  # (E, 3)
        radial = torch.sum(coord_diff ** 2, dim=1, keepdim=True)  # (E, 1)

        # Message passing
        h_new, coord_diff_scaled = self.propagate(
            edge_index,
            h=h,
            edge_attr=edge_attr,
            coord_diff=coord_diff,
            radial=radial
        )

        # Update node features
        h_new = h + h_new  # Residual connection

        # Update coordinates (if enabled)
        if self.update_coords:
            pos_new = pos + coord_diff_scaled
        else:
            pos_new = pos

        return h_new, pos_new

    def message(self, h_i, h_j, edge_attr, coord_diff, radial):
        """
        Construct messages from neighbors.

        Args:
            h_i: Node features of target nodes (E, node_dim)
            h_j: Node features of source nodes (E, node_dim)
            edge_attr: Edge attributes (E, edge_dim)
            coord_diff: Coordinate differences (E, 3)
            radial: Squared distances (E, 1)

        Returns:
            Messages for node feature update (E, hidden_dim)
            Scaled coordinate differences for position update (E, 3)
        """
        # Concatenate all information
        msg_input = torch.cat([h_i, h_j, edge_attr, radial], dim=-1)

        # Generate message
        msg = self.message_mlp(msg_input)  # (E, hidden_dim)

        # Coordinate update weights
        if self.update_coords:
            coord_weights = self.coord_mlp(msg)  # (E, 1)
            coord_diff_scaled = coord_diff * coord_weights  # (E, 3)
        else:
            coord_diff_scaled = torch.zeros_like(coord_diff)

        return msg, coord_diff_scaled

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Aggregate messages and coordinate updates separately.
        """
        msg, coord_diff_scaled = inputs

        # Aggregate messages (mean)
        msg_agg = super().aggregate(msg, index, ptr, dim_size)

        # Aggregate coordinate updates (sum)
        coord_agg = torch.zeros((dim_size or index.max() + 1, 3), device=coord_diff_scaled.device)
        coord_agg.index_add_(0, index, coord_diff_scaled)

        return msg_agg, coord_agg

    def update(self, aggr_out, h):
        """
        Update node features using aggregated messages.
        """
        msg_agg, coord_agg = aggr_out

        # Update node features
        h_input = torch.cat([h, msg_agg], dim=-1)
        h_update = self.node_mlp(h_input)

        return h_update, coord_agg


class EGNN(nn.Module):
    """
    Complete E(n)-Equivariant GNN for protein structure encoding.

    Processes 3D protein structures while preserving geometric equivariances.
    Uses ESM2 per-residue embeddings as initial node features.
    """

    def __init__(
        self,
        node_dim=1280,  # ESM2-650M output dimension
        hidden_dim=1280,
        num_layers=6,
        edge_embedding_dim=16,
        message_dim=24,
        dropout=0.1
    ):
        """
        Args:
            node_dim: Initial node feature dimension (ESM2 embeddings)
            hidden_dim: Hidden dimension for message passing
            num_layers: Number of EGNN layers
            edge_embedding_dim: Dimension for sinusoidal distance embeddings
            message_dim: Dimension for messages (paper uses 24)
            dropout: Dropout probability
        """
        super().__init__()

        self.node_dim = node_dim
        self.num_layers = num_layers

        # Distance embeddings
        self.distance_embedding = SinusoidalDistanceEmbedding(
            num_embeddings=edge_embedding_dim,
            theta=10000.0
        )

        # Initial node embedding projection (if needed)
        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        # EGNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            update_coords = (i < num_layers - 1)  # Don't update coords in last layer
            self.layers.append(
                EGNNLayer(
                    node_dim=hidden_dim,
                    edge_dim=edge_embedding_dim,
                    hidden_dim=message_dim,
                    update_coords=update_coords
                )
            )

        # Layer normalization
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, pos, edge_index):
        """
        Forward pass through EGNN.

        Args:
            h: Initial node features from ESM2 (N, node_dim)
            pos: 3D coordinates of Cα atoms (N, 3)
            edge_index: Edge connectivity from k-NN (2, E)

        Returns:
            h_out: Final node embeddings (N, hidden_dim)
            pos_out: Updated coordinates (N, 3)
        """
        # Compute edge attributes from distances
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        distances = torch.norm(coord_diff, dim=1)  # (E,)
        edge_attr = self.distance_embedding(distances)  # (E, edge_embedding_dim)

        # Initial node embedding
        h = self.node_embedding(h)

        # Pass through EGNN layers
        for layer, norm in zip(self.layers, self.norms):
            h_new, pos = layer(h, pos, edge_index, edge_attr)
            h = norm(h_new)
            h = self.dropout(h)

        return h, pos


class ProteinEncoderEGNN(nn.Module):
    """
    Complete protein encoder using EGNN for CLIPZyme.

    Pipeline:
    1. Extract ESM2 per-residue embeddings (1280-dim)
    2. Build k-NN graph from Cα coordinates
    3. Process with EGNN (6 layers, hidden=1280)
    4. Sum pooling for protein-level representation
    5. Project to embedding space and L2 normalize
    """

    def __init__(
        self,
        plm_name="facebook/esm2_t33_650M_UR50D",
        hidden_dim=1280,
        num_layers=6,
        proj_dim=512,
        dropout=0.1,
        k_neighbors=30,
        distance_cutoff=10.0
    ):
        """
        Args:
            plm_name: ESM2 model name (should be 650M for CLIPZyme)
            hidden_dim: Hidden dimension (1280 for CLIPZyme)
            num_layers: Number of EGNN layers (6 for CLIPZyme)
            proj_dim: Output embedding dimension
            dropout: Dropout probability
            k_neighbors: Number of nearest neighbors for graph construction
            distance_cutoff: Maximum distance for edges (Angstroms)
        """
        super().__init__()

        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(plm_name)
        self.esm2 = AutoModel.from_pretrained(plm_name)
        self.esm_hidden_dim = self.esm2.config.hidden_size

        # EGNN for structural processing
        self.egnn = EGNN(
            node_dim=self.esm_hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            edge_embedding_dim=16,
            message_dim=24,
            dropout=dropout
        )

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim)
        )

        self.k_neighbors = k_neighbors
        self.distance_cutoff = distance_cutoff

    def tokenize(self, sequences, max_len=650):
        """
        Tokenize protein sequences.

        Args:
            sequences: List of amino acid sequences
            max_len: Maximum sequence length (CLIPZyme uses 650)

        Returns:
            Tokenized batch
        """
        sequences = [seq[:max_len] for seq in sequences]
        return self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        )

    def build_knn_graph(self, coords, k=None, cutoff=None):
        """
        Build k-NN graph from Cα coordinates.

        Args:
            coords: Cα coordinates (N, 3)
            k: Number of nearest neighbors (default: self.k_neighbors)
            cutoff: Distance cutoff in Angstroms (default: self.distance_cutoff)

        Returns:
            edge_index: Edge connectivity (2, E)
        """
        k = k or self.k_neighbors
        cutoff = cutoff or self.distance_cutoff

        # Compute pairwise distances
        dists = torch.cdist(coords, coords)  # (N, N)

        # Get k nearest neighbors for each node
        _, indices = torch.topk(dists, k + 1, largest=False, dim=1)  # (N, k+1)
        indices = indices[:, 1:]  # Remove self-loops (N, k)

        # Build edge list
        src = torch.arange(coords.size(0), device=coords.device).unsqueeze(1).expand(-1, k)
        edge_index = torch.stack([src.flatten(), indices.flatten()], dim=0)

        # Apply distance cutoff
        row, col = edge_index
        edge_dists = torch.norm(coords[row] - coords[col], dim=1)
        mask = edge_dists <= cutoff
        edge_index = edge_index[:, mask]

        return edge_index

    @torch.no_grad()
    def extract_esm2_embeddings(self, batch_tokens):
        """
        Extract per-residue embeddings from ESM2.

        Args:
            batch_tokens: Tokenized sequences

        Returns:
            embeddings: Per-residue embeddings (B, L, hidden_dim)
            attention_mask: Mask for valid residues
        """
        outputs = self.esm2(**batch_tokens)
        return outputs.last_hidden_state, batch_tokens['attention_mask']

    def forward(self, batch_tokens, coords_list):
        """
        Forward pass through the protein encoder.

        Args:
            batch_tokens: Tokenized sequences (dict with input_ids, attention_mask)
            coords_list: List of Cα coordinate tensors [(N1, 3), (N2, 3), ...]
                        Each tensor has coordinates for one protein

        Returns:
            embeddings: L2-normalized protein embeddings (B, proj_dim)
        """
        device = next(self.parameters()).device

        # Extract ESM2 embeddings
        esm_embeddings, mask = self.extract_esm2_embeddings(batch_tokens)

        # Process each protein separately (different structures)
        batch_embeddings = []

        for i, coords in enumerate(coords_list):
            coords = coords.to(device)

            # Get embeddings for this protein (remove special tokens)
            h = esm_embeddings[i, 1:len(coords)+1]  # (N, hidden_dim)

            # Build k-NN graph
            edge_index = self.build_knn_graph(coords)

            # Process with EGNN
            h_out, _ = self.egnn(h, coords, edge_index)

            # Sum pooling (as in CLIPZyme)
            g = torch.sum(h_out, dim=0, keepdim=True)  # (1, hidden_dim)

            batch_embeddings.append(g)

        # Stack batch
        g = torch.cat(batch_embeddings, dim=0)  # (B, hidden_dim)

        # Project and normalize
        z = self.projection(g)
        z = F.normalize(z, p=2, dim=-1)

        return z

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.projection.net[-1].out_features
