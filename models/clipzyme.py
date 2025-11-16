"""
Unified CLIPZyme model.

Combines protein and reaction encoders for contrastive learning.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from common.interfaces import ProteinEncoder, ReactionEncoder
from reaction_encoder.loss import clip_loss


class CLIPZymeModel(nn.Module):
    """
    Complete CLIPZyme model for protein-reaction matching.

    Combines protein and reaction encoders with CLIP loss for
    contrastive learning in a shared embedding space.
    """

    def __init__(
        self,
        protein_encoder: ProteinEncoder,
        reaction_encoder: ReactionEncoder,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
    ):
        """
        Initialize CLIPZyme model.

        Args:
            protein_encoder: Encoder for protein sequences/structures
            reaction_encoder: Encoder for chemical reactions
            temperature: Temperature parameter for CLIP loss
            learnable_temperature: Whether temperature is a learnable parameter
        """
        super().__init__()

        self.protein_encoder = protein_encoder
        self.reaction_encoder = reaction_encoder

        # Temperature parameter
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))

        self._learnable_temperature = learnable_temperature

    def forward(
        self,
        protein_inputs: Dict[str, Any],
        reaction_inputs: Any,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both encoders.

        Args:
            protein_inputs: Dictionary of protein encoder inputs
            reaction_inputs: Reaction encoder inputs (graph data)
            return_embeddings: If True, return embeddings instead of loss

        Returns:
            Dictionary containing:
                - loss: CLIP loss (if return_embeddings=False)
                - protein_embeddings: Protein embeddings (B, D)
                - reaction_embeddings: Reaction embeddings (B, D)
                - logits_per_protein: Similarity matrix from protein view (B, B)
                - logits_per_reaction: Similarity matrix from reaction view (B, B)
        """
        # Encode proteins
        protein_embeddings = self.protein_encoder(**protein_inputs)

        # Encode reactions
        reaction_embeddings = self.reaction_encoder(reaction_inputs)

        # Compute similarities
        logits_per_protein = protein_embeddings @ reaction_embeddings.t() / self.temperature
        logits_per_reaction = logits_per_protein.t()

        output = {
            'protein_embeddings': protein_embeddings,
            'reaction_embeddings': reaction_embeddings,
            'logits_per_protein': logits_per_protein,
            'logits_per_reaction': logits_per_reaction,
        }

        if not return_embeddings:
            # Compute CLIP loss
            loss = clip_loss(protein_embeddings, reaction_embeddings, self.temperature.item())
            output['loss'] = loss

        return output

    def encode_proteins(
        self,
        sequences: list,
        device: str = "cpu",
        **kwargs
    ) -> torch.Tensor:
        """
        Encode protein sequences.

        Args:
            sequences: List of protein sequences
            device: Device to use
            **kwargs: Additional arguments for encoder

        Returns:
            Protein embeddings (N, D)
        """
        return self.protein_encoder.encode(sequences, device=device, **kwargs)

    def encode_reactions(
        self,
        reactions: list,
        device: str = "cpu",
        **kwargs
    ) -> torch.Tensor:
        """
        Encode chemical reactions.

        Args:
            reactions: List of reaction SMILES
            device: Device to use
            **kwargs: Additional arguments for encoder

        Returns:
            Reaction embeddings (N, D)
        """
        return self.reaction_encoder.encode(reactions, device=device, **kwargs)

    def compute_similarity(
        self,
        proteins: list,
        reactions: list,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Compute protein-reaction similarity matrix.

        Args:
            proteins: List of protein sequences
            reactions: List of reaction SMILES
            device: Device to use

        Returns:
            Similarity matrix (n_proteins, n_reactions)
        """
        protein_emb = self.encode_proteins(proteins, device=device)
        reaction_emb = self.encode_reactions(reactions, device=device)

        return protein_emb @ reaction_emb.t()

    def get_embedding_dim(self) -> int:
        """Get the dimension of the shared embedding space."""
        return self.protein_encoder.get_embedding_dim()

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.parameters()).device

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns:
            Dictionary with model configuration
        """
        return {
            'protein_encoder_type': type(self.protein_encoder).__name__,
            'reaction_encoder_type': type(self.reaction_encoder).__name__,
            'embedding_dim': self.get_embedding_dim(),
            'temperature': self.temperature.item(),
            'learnable_temperature': self._learnable_temperature,
        }

    def save_pretrained(self, save_path: str):
        """
        Save model weights and configuration.

        Args:
            save_path: Path to save directory
        """
        import os
        import json

        os.makedirs(save_path, exist_ok=True)

        # Save weights
        torch.save(self.state_dict(), os.path.join(save_path, 'model.pt'))

        # Save config
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(self.get_config(), f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        load_path: str,
        protein_encoder: ProteinEncoder,
        reaction_encoder: ReactionEncoder
    ) -> 'CLIPZymeModel':
        """
        Load pretrained model.

        Args:
            load_path: Path to saved model directory
            protein_encoder: Initialized protein encoder
            reaction_encoder: Initialized reaction encoder

        Returns:
            Loaded CLIPZymeModel
        """
        import os
        import json

        # Load config
        with open(os.path.join(load_path, 'config.json'), 'r') as f:
            config = json.load(f)

        # Create model
        model = cls(
            protein_encoder=protein_encoder,
            reaction_encoder=reaction_encoder,
            temperature=config['temperature'],
            learnable_temperature=config['learnable_temperature'],
        )

        # Load weights
        model.load_state_dict(torch.load(os.path.join(load_path, 'model.pt')))

        return model


__all__ = ['CLIPZymeModel']
