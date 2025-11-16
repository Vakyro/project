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

        Args:
            config: ProteinEncoderConfig or dict with configuration

        Returns:
            ProteinEncoder instance

        Raises:
            ValueError: If encoder type is not recognized
        """
        # Convert dict to config if needed
        if isinstance(config, dict):
            config = ProteinEncoderConfig(**config)

        encoder_type = config.type

        if encoder_type == 'ESM2':
            return EncoderFactory._create_esm2_encoder(config)
        elif encoder_type == 'EGNN':
            return EncoderFactory._create_egnn_encoder(config)
        else:
            raise ValueError(
                f"Unknown protein encoder type: {encoder_type}. "
                f"Supported types: 'ESM2', 'EGNN'"
            )

    @staticmethod
    def _create_esm2_encoder(config: ProteinEncoderConfig) -> ProteinEncoder:
        """Create ESM2-based protein encoder."""
        from protein_encoder.esm_model import ProteinEncoderESM2

        return ProteinEncoderESM2(
            plm_name=config.plm_name,
            pooling=config.pooling,
            proj_dim=config.proj_dim,
            dropout=config.dropout,
            gradient_checkpointing=config.gradient_checkpointing,
        )

    @staticmethod
    def _create_egnn_encoder(config: ProteinEncoderConfig) -> ProteinEncoder:
        """Create EGNN-based protein encoder (ESM2 + E(n)-GNN)."""
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

        Args:
            config: ReactionEncoderConfig or dict with configuration

        Returns:
            ReactionEncoder instance

        Raises:
            ValueError: If encoder type is not recognized
        """
        # Convert dict to config if needed
        if isinstance(config, dict):
            config = ReactionEncoderConfig(**config)

        encoder_type = config.type

        if encoder_type == 'GNN':
            return EncoderFactory._create_basic_gnn_encoder(config)
        elif encoder_type == 'DMPNN':
            return EncoderFactory._create_dmpnn_encoder(config)
        else:
            raise ValueError(
                f"Unknown reaction encoder type: {encoder_type}. "
                f"Supported types: 'GNN', 'DMPNN'"
            )

    @staticmethod
    def _get_feature_dimensions(config: ReactionEncoderConfig):
        """Determine feature dimensions based on feature type."""
        from common.constants import ChemistryConfig

        if config.feature_type == 'basic':
            base_node_dim = ChemistryConfig.ATOM_FEATURE_DIM_BASIC
            edge_dim = ChemistryConfig.EDGE_FEATURE_DIM_BASIC
        else:  # clipzyme
            base_node_dim = ChemistryConfig.ATOM_FEATURE_DIM_CLIPZYME
            edge_dim = ChemistryConfig.EDGE_FEATURE_DIM_CLIPZYME

        # Node features depend on use_enhanced_features:
        # - Enhanced: reactant (base) + product (base) + existence (2) + one-hot (11) + reactive (1)
        #            = 2*base + 14
        # - Not enhanced: reactant (base) + product (base) + existence (2) + changed (1)
        #                = 2*base + 3
        # E.g., for basic (base=7):
        #   - Enhanced: 7 + 7 + 2 + 11 + 1 = 28
        #   - Not enhanced: 7 + 7 + 2 + 1 = 17
        if config.use_enhanced_features:
            node_dim = 2 * base_node_dim + 14
        else:
            node_dim = 2 * base_node_dim + 3

        # Override if explicitly set
        if config.node_dim is not None:
            node_dim = config.node_dim
        if config.edge_dim is not None:
            edge_dim = config.edge_dim

        return node_dim, edge_dim

    @staticmethod
    def _create_basic_gnn_encoder(config: ReactionEncoderConfig):
        """Create basic GNN encoder."""
        from reaction_encoder.model import ReactionGNN
        from common.reaction_encoder_wrapper import ReactionEncoderWrapper

        node_dim, edge_dim = EncoderFactory._get_feature_dimensions(config)

        model = ReactionGNN(
            x_dim=node_dim,
            e_dim=edge_dim,
            hidden=config.hidden_dim,
            layers=config.num_layers,
            out_dim=config.proj_dim,
        )

        return ReactionEncoderWrapper(
            model=model,
            feature_type=config.feature_type,
            use_dual_branch=False,
            use_enhanced_features=config.use_enhanced_features
        )

    @staticmethod
    def _create_dmpnn_encoder(config: ReactionEncoderConfig):
        """Create Two-Stage DMPNN encoder (CLIPZyme architecture)."""
        from reaction_encoder.dmpnn import TwoStageDMPNN
        from common.reaction_encoder_wrapper import ReactionEncoderWrapper

        node_dim, edge_dim = EncoderFactory._get_feature_dimensions(config)

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
            feature_type=config.feature_type,
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
        elif config == 'faithful':
            full_config = CLIPZymeConfig.clipzyme_faithful()
        else:
            raise ValueError(f"Unknown preset: {config}")
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
        elif config == 'faithful':
            full_config = CLIPZymeConfig.clipzyme_faithful()
        else:
            raise ValueError(f"Unknown preset: {config}")
        config = full_config.reaction_encoder

    return EncoderFactory.create_reaction_encoder(config)


def create_model(config: Union[CLIPZymeConfig, str]) -> nn.Module:
    """
    Create complete CLIPZyme model.

    Args:
        config: Configuration (CLIPZymeConfig or preset name)

    Returns:
        CLIPZyme model
    """
    if isinstance(config, str):
        # Load preset or from YAML
        from config.config import CLIPZymeConfig, load_config
        if config == 'default':
            config = CLIPZymeConfig.default()
        elif config == 'faithful':
            config = CLIPZymeConfig.clipzyme_faithful()
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
