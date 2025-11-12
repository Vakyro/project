"""
Abstract base classes and interfaces for CLIPZyme encoders.

Defines the contract that all encoders must implement, enabling
the Strategy Pattern for easy encoder substitution.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import torch


class ProteinEncoder(ABC):
    """
    Abstract base class for protein encoders.

    All protein encoder implementations must inherit from this class
    and implement its abstract methods.
    """

    @abstractmethod
    def encode(
        self,
        sequences: Union[List[str], str],
        device: str = "cpu",
        batch_size: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode protein sequence(s) to embedding vector(s).

        Args:
            sequences: Single sequence or list of sequences
            device: Device to run encoding on ('cpu' or 'cuda')
            batch_size: Batch size for processing (if None, process all at once)
            **kwargs: Additional encoder-specific arguments

        Returns:
            Tensor of shape (N, D) where N is number of sequences
            and D is embedding dimension. Embeddings are L2-normalized.
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of the output embeddings.

        Returns:
            Integer dimension of embedding vectors
        """
        pass

    @abstractmethod
    def tokenize(self, sequences: List[str], max_len: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize sequences for model input.

        Args:
            sequences: List of protein sequences
            max_len: Maximum sequence length (truncate if longer)

        Returns:
            Dictionary with tokenized inputs (input_ids, attention_mask, etc.)
        """
        pass

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the encoder.

        This is typically called by encode() after tokenization.
        """
        raise NotImplementedError("Subclasses should implement forward()")

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Get the device the model is on."""
        pass

    def to(self, device: Union[str, torch.device]):
        """Move model to device."""
        raise NotImplementedError("Subclasses should implement to()")

    def eval(self):
        """Set model to evaluation mode."""
        raise NotImplementedError("Subclasses should implement eval()")

    def train(self, mode: bool = True):
        """Set model to training mode."""
        raise NotImplementedError("Subclasses should implement train()")


class ReactionEncoder(ABC):
    """
    Abstract base class for reaction encoders.

    All reaction encoder implementations must inherit from this class
    and implement its abstract methods.
    """

    @abstractmethod
    def encode(
        self,
        reactions: Union[List[str], str],
        device: str = "cpu",
        batch_size: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode chemical reaction(s) to embedding vector(s).

        Args:
            reactions: Single SMILES or list of SMILES strings
            device: Device to run encoding on ('cpu' or 'cuda')
            batch_size: Batch size for processing
            **kwargs: Additional encoder-specific arguments

        Returns:
            Tensor of shape (N, D) where N is number of reactions
            and D is embedding dimension. Embeddings are L2-normalized.
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of the output embeddings.

        Returns:
            Integer dimension of embedding vectors
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args depend on the specific encoder implementation.
        Typically takes graph data structures.
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Get the device the model is on."""
        pass

    def to(self, device: Union[str, torch.device]):
        """Move model to device."""
        raise NotImplementedError("Subclasses should implement to()")

    def eval(self):
        """Set model to evaluation mode."""
        raise NotImplementedError("Subclasses should implement eval()")

    def train(self, mode: bool = True):
        """Set model to training mode."""
        raise NotImplementedError("Subclasses should implement train()")


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extraction.

    Enables Strategy Pattern for different feature extraction methods.
    """

    @abstractmethod
    def extract_atom_features(self, atom) -> List[float]:
        """
        Extract features from an RDKit atom.

        Args:
            atom: RDKit Atom object

        Returns:
            List of numerical features
        """
        pass

    @abstractmethod
    def extract_bond_features(self, bond) -> List[float]:
        """
        Extract features from an RDKit bond.

        Args:
            bond: RDKit Bond object

        Returns:
            List of numerical features
        """
        pass

    @abstractmethod
    def get_atom_feature_dim(self) -> int:
        """Get dimensionality of atom features."""
        pass

    @abstractmethod
    def get_bond_feature_dim(self) -> int:
        """Get dimensionality of bond features."""
        pass


class DataRepository(ABC):
    """
    Abstract base class for data repositories.

    Implements Repository Pattern for data access.
    """

    @abstractmethod
    def load_all(self, **filters) -> List[Any]:
        """Load all items matching filters."""
        pass

    @abstractmethod
    def load_by_id(self, item_id: str) -> Any:
        """Load single item by ID."""
        pass

    @abstractmethod
    def count(self, **filters) -> int:
        """Count items matching filters."""
        pass


# Export all interfaces
__all__ = [
    'ProteinEncoder',
    'ReactionEncoder',
    'FeatureExtractor',
    'DataRepository',
]
