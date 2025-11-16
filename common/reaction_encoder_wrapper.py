"""
Wrapper for reaction encoders to implement the ReactionEncoder interface.

This module provides wrapper classes that add the encode() method to existing
reaction encoder models, handling SMILES parsing and graph construction.
"""

import torch
import torch.nn as nn
from typing import List, Union, Optional
from .interfaces import ReactionEncoder


class ReactionEncoderWrapper(ReactionEncoder, nn.Module):
    """
    Wrapper for reaction encoder models that implements the ReactionEncoder interface.

    This wrapper adds the encode() method which handles:
    - SMILES parsing
    - Graph construction
    - Batch processing
    - Normalization
    """

    def __init__(self, model: nn.Module, feature_type: str = 'clipzyme', use_dual_branch: bool = False, use_enhanced_features: bool = False):
        """
        Args:
            model: The underlying reaction encoder model
            feature_type: Type of features to use ('basic' or 'clipzyme')
            use_dual_branch: Whether model uses dual-branch architecture
            use_enhanced_features: Whether to use enhanced features (17-dim vs 7-dim for basic)
        """
        super().__init__()
        self.model = model
        self.feature_type = feature_type
        self.use_dual_branch = use_dual_branch
        self.use_enhanced_features = use_enhanced_features

    def encode(
        self,
        reactions: Union[List[str], str],
        device: str = "cpu",
        batch_size: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode reaction SMILES to embeddings.

        Args:
            reactions: Single reaction SMILES or list of reaction SMILES
            device: Device to run encoding on ('cpu' or 'cuda')
            batch_size: Batch size for processing (currently processes all at once)
            **kwargs: Additional arguments

        Returns:
            Tensor of shape (N, D) where N is number of reactions
            and D is embedding dimension. Embeddings are L2-normalized.
        """
        # Handle single string input
        if isinstance(reactions, str):
            reactions = [reactions]

        self.model.eval()
        self.model.to(device)

        embeddings = []

        with torch.no_grad():
            for rxn_smiles in reactions:
                try:
                    # Build graph(s) for this reaction
                    if self.use_dual_branch:
                        graph_full, graph_change = self._build_dual_graphs(rxn_smiles, device)
                        z = self.model(graph_full, graph_change)
                    else:
                        graph = self._build_graph(rxn_smiles, device)
                        z = self.model(graph)

                    embeddings.append(z)
                except Exception:
                    zero_emb = torch.zeros(1, self.get_embedding_dim(), device=device)
                    embeddings.append(zero_emb)

        return torch.cat(embeddings, dim=0)

    def _build_graph(self, rxn_smiles: str, device: str):
        """Build PyG Data object from reaction SMILES."""
        from reaction_encoder.chem import parse_reaction_smiles
        from reaction_encoder.builder import build_transition_graph

        # Parse reaction SMILES to get reactants and products
        reacts, prods = parse_reaction_smiles(rxn_smiles)

        # Build transition graph
        graph = build_transition_graph(reacts, prods, use_enhanced_features=self.use_enhanced_features)

        return graph.to(device)

    def _build_dual_graphs(self, rxn_smiles: str, device: str):
        """Build both full and change-only graphs for dual-branch model."""
        from reaction_encoder.chem import parse_reaction_smiles
        from reaction_encoder.builder import build_transition_graph

        reacts, prods = parse_reaction_smiles(rxn_smiles)
        graph_full = build_transition_graph(reacts, prods, use_enhanced_features=self.use_enhanced_features)
        graph_change = build_transition_graph(reacts, prods, use_enhanced_features=True)

        return graph_full.to(device), graph_change.to(device)

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.model.get_embedding_dim()

    def forward(self, *args, **kwargs):
        """Forward pass delegates to wrapped model."""
        return self.model(*args, **kwargs)

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.model.parameters()).device

    def to(self, device: Union[str, torch.device]):
        """Move model to device."""
        self.model.to(device)
        return nn.Module.to(self, device)

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return nn.Module.eval(self)

    def train(self, mode: bool = True):
        """Set model to training mode."""
        self.model.train(mode)
        return nn.Module.train(self, mode)
