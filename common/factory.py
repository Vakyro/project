"""
Factory pattern for creating encoders and models.

Enables dynamic instantiation based on configuration,
supporting the Strategy pattern for encoder selection.
"""

from typing import Union, Dict, Any
import torch.nn as nn

from common.interfaces import ProteinEncoder, ReactionEncoder
from config.config import (
    ProteinEncoderConfig,
    ReactionEncoderConfig,
    CLIPZymeConfig
)


class EncoderFactory:
    """
    Factory for creating protein and reaction encoders.

    Uses configuration to instantiate the appropriate encoder type.
    """

    @staticmethod
    def create_protein_encoder(
        config: Union[ProteinEncoderConfig, Dict[str, Any]]
    ) -> ProteinEncoder:
        """
        Create protein encoder from configuration.

        Production implementation uses ProteinEncoderEGNN (ESM2 + E(n)-equivariant GNN)
        matching the CLIPZyme paper exactly.

        Args:
            config: ProteinEncoderConfig or dict with configuration

        Returns:
            ProteinEncoder instance (ProteinEncoderEGNN)

        Raises:
            ValueError: If encoder type is not 'EGNN'
        """
        # Convert dict to config if needed
        if isinstance(config, dict):
            config = ProteinEncoderConfig(**config)

        encoder_type = config.type

        if encoder_type == 'EGNN':
            return EncoderFactory._create_egnn_encoder(config)
        else:
            raise ValueError(
                f"Unknown protein encoder type: {encoder_type}. "
                f"Only 'EGNN' (ESM2 + E(n)-GNN) is supported in production."
            )

    @staticmethod
    def _create_egnn_encoder(config: ProteinEncoderConfig) -> ProteinEncoder:
        """
        Create EGNN-based protein encoder (ESM2 + E(n)-equivariant GNN).

        This is the CLIPZyme paper architecture.
        """
        from protein_encoder.egnn import ProteinEncoderEGNN

        return ProteinEncoderEGNN(
            plm_name=config.plm_name,
            hidden_dim=config.egnn_hidden_dim,
            num_layers=config.egnn_layers,
            proj_dim=config.proj_dim,
            dropout=config.dropout,
            k_neighbors=config.k_neighbors,
            distance_cutoff=config.distance_cutoff,
        )

    @staticmethod
    def create_reaction_encoder(
        config: Union[ReactionEncoderConfig, Dict[str, Any]]
    ) -> ReactionEncoder:
        """
        Create reaction encoder from configuration.

        Production implementation uses TwoStageDMPNN (Directed Message Passing Neural Network)
        matching the CLIPZyme paper exactly.

        Args:
            config: ReactionEncoderConfig or dict with configuration

        Returns:
            ReactionEncoder instance (TwoStageDMPNN)

        Raises:
            ValueError: If encoder type is not 'DMPNN'
        """
        # Convert dict to config if needed
        if isinstance(config, dict):
            config = ReactionEncoderConfig(**config)

        encoder_type = config.type

        if encoder_type == 'DMPNN':
            return EncoderFactory._create_dmpnn_encoder(config)
        else:
            raise ValueError(
                f"Unknown reaction encoder type: {encoder_type}. "
                f"Only 'DMPNN' (Two-Stage Directed MPNN) is supported in production."
            )

    @staticmethod
    def _create_dmpnn_encoder(config: ReactionEncoderConfig):
        """
        Create Two-Stage DMPNN encoder (CLIPZyme architecture).

        Uses CLIPZyme features (9 atom features, 3 edge features) as specified in the paper.
        """
        from reaction_encoder.dmpnn import TwoStageDMPNN
        from common.reaction_encoder_wrapper import ReactionEncoderWrapper
        from common.constants import ChemistryConfig

        # Production uses CLIPZyme features only
        node_dim = config.node_dim if config.node_dim is not None else ChemistryConfig.ATOM_FEATURE_DIM_CLIPZYME
        edge_dim = config.edge_dim if config.edge_dim is not None else ChemistryConfig.EDGE_FEATURE_DIM_CLIPZYME

        model = TwoStageDMPNN(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=config.dmpnn_hidden_dim,
            num_layers=config.num_layers,
            proj_dim=config.proj_dim,
            dropout=config.dropout,
        )

        return ReactionEncoderWrapper(
            model=model,
            feature_type='clipzyme',  # Production always uses CLIPZyme features
            use_dual_branch=False,
            use_enhanced_features=config.use_enhanced_features
        )

class ModelFactory:
    """
    Factory for creating complete CLIPZyme models.

    Combines protein and reaction encoders into a unified model.
    """

    @staticmethod
    def create_clipzyme_model(config: CLIPZymeConfig) -> nn.Module:
        """
        Create complete CLIPZyme model from configuration.

        Args:
            config: CLIPZymeConfig object

        Returns:
            CLIPZyme model (nn.Module)
        """
        from models.clipzyme import CLIPZymeModel

        # Create encoders
        protein_encoder = EncoderFactory.create_protein_encoder(config.protein_encoder)
        reaction_encoder = EncoderFactory.create_reaction_encoder(config.reaction_encoder)

        # Create combined model
        model = CLIPZymeModel(
            protein_encoder=protein_encoder,
            reaction_encoder=reaction_encoder,
            temperature=config.training.temperature,
            learnable_temperature=config.training.learnable_temperature,
        )

        return model


# Convenience functions
def create_protein_encoder(config: Union[ProteinEncoderConfig, Dict, str]) -> ProteinEncoder:
    """
    Create protein encoder.

    Args:
        config: Configuration (ProteinEncoderConfig, dict, or preset name)

    Returns:
        ProteinEncoder instance
    """
    if isinstance(config, str):
        # Load preset configuration
        from config.config import CLIPZymeConfig
        if config == 'default':
            full_config = CLIPZymeConfig.default()
        else:
            raise ValueError(f"Unknown preset: {config}. Only 'default' is supported.")
        config = full_config.protein_encoder

    return EncoderFactory.create_protein_encoder(config)


def create_reaction_encoder(config: Union[ReactionEncoderConfig, Dict, str]) -> ReactionEncoder:
    """
    Create reaction encoder.

    Args:
        config: Configuration (ReactionEncoderConfig, dict, or preset name)

    Returns:
        ReactionEncoder instance
    """
    if isinstance(config, str):
        # Load preset configuration
        from config.config import CLIPZymeConfig
        if config == 'default':
            full_config = CLIPZymeConfig.default()
        else:
            raise ValueError(f"Unknown preset: {config}. Only 'default' is supported.")
        config = full_config.reaction_encoder

    return EncoderFactory.create_reaction_encoder(config)


def create_model(config: Union[CLIPZymeConfig, str]) -> nn.Module:
    """
    Create complete CLIPZyme model.

    Args:
        config: Configuration (CLIPZymeConfig or preset name / YAML path)

    Returns:
        CLIPZyme model
    """
    if isinstance(config, str):
        # Load preset or from YAML
        from config.config import CLIPZymeConfig, load_config
        if config == 'default':
            config = CLIPZymeConfig.default()
        else:
            # Assume it's a path to YAML file
            config = load_config(config)

    return ModelFactory.create_clipzyme_model(config)


# Export
__all__ = [
    'EncoderFactory',
    'ModelFactory',
    'create_protein_encoder',
    'create_reaction_encoder',
    'create_model',
]
