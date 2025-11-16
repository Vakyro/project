"""
Builder pattern for constructing CLIPZyme models.

Provides fluent API for model construction.
"""

from typing import Optional, Union
import torch

from common.interfaces import ProteinEncoder, ReactionEncoder
from common.factory import EncoderFactory
from config.config import (
    CLIPZymeConfig,
    ProteinEncoderConfig,
    ReactionEncoderConfig,
)
from .clipzyme import CLIPZymeModel


class CLIPZymeBuilder:
    """
    Builder for CLIPZyme models.

    Provides fluent interface for constructing models step-by-step.

    Example:
        >>> model = (CLIPZymeBuilder()
        ...          .with_protein_encoder_config(protein_config)
        ...          .with_reaction_encoder_config(reaction_config)
        ...          .with_temperature(0.07)
        ...          .on_device('cuda')
        ...          .build())
    """

    def __init__(self):
        """Initialize builder with default settings."""
        self._protein_encoder: Optional[ProteinEncoder] = None
        self._reaction_encoder: Optional[ReactionEncoder] = None
        self._protein_config: Optional[ProteinEncoderConfig] = None
        self._reaction_config: Optional[ReactionEncoderConfig] = None
        self._temperature: float = 0.07
        self._learnable_temperature: bool = False
        self._device: Union[str, torch.device] = 'cpu'

    def with_protein_encoder(self, encoder: ProteinEncoder) -> 'CLIPZymeBuilder':
        """
        Set protein encoder directly.

        Args:
            encoder: Initialized ProteinEncoder

        Returns:
            Self for chaining
        """
        self._protein_encoder = encoder
        return self

    def with_reaction_encoder(self, encoder: ReactionEncoder) -> 'CLIPZymeBuilder':
        """
        Set reaction encoder directly.

        Args:
            encoder: Initialized ReactionEncoder

        Returns:
            Self for chaining
        """
        self._reaction_encoder = encoder
        return self

    def with_protein_encoder_config(
        self,
        config: Union[ProteinEncoderConfig, dict]
    ) -> 'CLIPZymeBuilder':
        """
        Set protein encoder configuration.

        The encoder will be created during build().

        Args:
            config: ProteinEncoderConfig or dict

        Returns:
            Self for chaining
        """
        if isinstance(config, dict):
            config = ProteinEncoderConfig(**config)
        self._protein_config = config
        return self

    def with_reaction_encoder_config(
        self,
        config: Union[ReactionEncoderConfig, dict]
    ) -> 'CLIPZymeBuilder':
        """
        Set reaction encoder configuration.

        The encoder will be created during build().

        Args:
            config: ReactionEncoderConfig or dict

        Returns:
            Self for chaining
        """
        if isinstance(config, dict):
            config = ReactionEncoderConfig(**config)
        self._reaction_config = config
        return self

    def with_config(self, config: CLIPZymeConfig) -> 'CLIPZymeBuilder':
        """
        Set complete CLIPZyme configuration.

        Args:
            config: CLIPZymeConfig object

        Returns:
            Self for chaining
        """
        self._protein_config = config.protein_encoder
        self._reaction_config = config.reaction_encoder
        self._temperature = config.training.temperature
        self._learnable_temperature = config.training.learnable_temperature
        self._device = config.training.device
        return self

    def with_temperature(self, temperature: float) -> 'CLIPZymeBuilder':
        """
        Set CLIP temperature parameter.

        Args:
            temperature: Temperature value (typical: 0.01-0.1)

        Returns:
            Self for chaining
        """
        self._temperature = temperature
        return self

    def with_learnable_temperature(self, learnable: bool = True) -> 'CLIPZymeBuilder':
        """
        Set whether temperature is learnable.

        Args:
            learnable: If True, temperature is a trainable parameter

        Returns:
            Self for chaining
        """
        self._learnable_temperature = learnable
        return self

    def on_device(self, device: Union[str, torch.device]) -> 'CLIPZymeBuilder':
        """
        Set device for model.

        Args:
            device: Device string ('cpu', 'cuda') or torch.device

        Returns:
            Self for chaining
        """
        self._device = device
        return self

    def from_preset(self, preset: str) -> 'CLIPZymeBuilder':
        """
        Load configuration from preset.

        Args:
            preset: Preset name ('default', 'faithful')

        Returns:
            Self for chaining
        """
        if preset == 'default':
            config = CLIPZymeConfig.default()
        elif preset == 'faithful':
            config = CLIPZymeConfig.clipzyme_faithful()
        else:
            raise ValueError(f"Unknown preset: {preset}")

        return self.with_config(config)

    def from_yaml(self, yaml_path: str) -> 'CLIPZymeBuilder':
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration

        Returns:
            Self for chaining
        """
        config = CLIPZymeConfig.from_yaml(yaml_path)
        return self.with_config(config)

    def build(self) -> CLIPZymeModel:
        """
        Build the CLIPZyme model.

        Creates encoders from configurations if not already set,
        then constructs the final model.

        Returns:
            Initialized CLIPZymeModel

        Raises:
            ValueError: If neither encoders nor configs are set
        """
        # Create protein encoder if needed
        if self._protein_encoder is None:
            if self._protein_config is None:
                raise ValueError(
                    "Either protein_encoder or protein_encoder_config must be set"
                )
            self._protein_encoder = EncoderFactory.create_protein_encoder(
                self._protein_config
            )

        # Create reaction encoder if needed
        if self._reaction_encoder is None:
            if self._reaction_config is None:
                raise ValueError(
                    "Either reaction_encoder or reaction_encoder_config must be set"
                )
            self._reaction_encoder = EncoderFactory.create_reaction_encoder(
                self._reaction_config
            )

        # Create model
        model = CLIPZymeModel(
            protein_encoder=self._protein_encoder,
            reaction_encoder=self._reaction_encoder,
            temperature=self._temperature,
            learnable_temperature=self._learnable_temperature,
        )

        # Move to device
        model = model.to(self._device)

        return model

    def reset(self) -> 'CLIPZymeBuilder':
        """
        Reset builder to initial state.

        Returns:
            Self for chaining
        """
        self.__init__()
        return self


# Convenience function
def build_clipzyme_model(
    config: Optional[Union[CLIPZymeConfig, str]] = None,
    device: str = 'cpu'
) -> CLIPZymeModel:
    """
    Build CLIPZyme model from configuration.

    Args:
        config: CLIPZymeConfig, preset name, or YAML path
                If None, uses default configuration
        device: Device to place model on

    Returns:
        Initialized CLIPZymeModel

    Examples:
        >>> # From default config
        >>> model = build_clipzyme_model()

        >>> # From preset
        >>> model = build_clipzyme_model('faithful', device='cuda')

        >>> # From YAML
        >>> model = build_clipzyme_model('configs/my_config.yaml')

        >>> # From config object
        >>> config = CLIPZymeConfig.default()
        >>> model = build_clipzyme_model(config, device='cuda')
    """
    builder = CLIPZymeBuilder().on_device(device)

    if config is None:
        # Use default
        builder = builder.from_preset('default')
    elif isinstance(config, str):
        # Check if it's a preset or file path
        if config in ['default', 'faithful']:
            builder = builder.from_preset(config)
        else:
            # Assume it's a YAML path
            builder = builder.from_yaml(config)
    elif isinstance(config, CLIPZymeConfig):
        builder = builder.with_config(config)
    else:
        raise ValueError(f"Invalid config type: {type(config)}")

    return builder.build()


__all__ = ['CLIPZymeBuilder', 'build_clipzyme_model']
