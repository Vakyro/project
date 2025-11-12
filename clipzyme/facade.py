"""
Facade pattern for CLIPZyme.

Provides simplified, high-level API for common use cases.
"""

from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import torch
import numpy as np

from config.config import CLIPZymeConfig, load_config
from models.builder import build_clipzyme_model
from models.clipzyme import CLIPZymeModel
from data.repositories import (
    ProteinRepository,
    ReactionRepository,
    EnzymeReactionRepository,
)
from common.constants import DataConfig


class CLIPZyme:
    """
    High-level API for CLIPZyme.

    Simplifies common operations:
    - Encoding proteins and reactions
    - Computing similarities
    - Finding best matches
    - Loading data

    Example:
        >>> # Quick start with defaults
        >>> clipzyme = CLIPZyme()
        >>>
        >>> # Encode and compare
        >>> proteins = ["MSKGEEL...", "MAHHHHH..."]
        >>> reactions = ["[N:1]=[N:2]>>[N:1][N:2]", "[C:1]=[C:2]>>[C:1][C:2]"]
        >>> similarity = clipzyme.compute_similarity(proteins, reactions)
        >>>
        >>> # Find best matches
        >>> best_reactions = clipzyme.find_best_reactions_for_protein(proteins[0], reactions)
    """

    def __init__(
        self,
        config: Optional[Union[str, CLIPZymeConfig]] = None,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize CLIPZyme facade.

        Args:
            config: Configuration (YAML path, preset name, or CLIPZymeConfig)
                    Options: 'default', 'faithful', or path to YAML file
                    If None, uses default configuration
            device: Device to use ('cpu' or 'cuda'). Auto-detects if None
            model_path: Path to pretrained model (optional)
        """
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load configuration
        if config is None:
            config = 'default'

        if isinstance(config, str):
            if config in ['default', 'faithful']:
                # Load preset
                if config == 'default':
                    self.config = CLIPZymeConfig.default()
                else:
                    self.config = CLIPZymeConfig.clipzyme_faithful()
            else:
                # Load from YAML
                self.config = load_config(config)
        else:
            self.config = config

        # Override device in config
        self.config.training.device = device
        self.device = device

        # Build model
        self.model = build_clipzyme_model(self.config, device=device)
        self.model.eval()

        # Load pretrained weights if provided
        if model_path is not None:
            self.load_pretrained(model_path)

        # Initialize data repositories
        self._protein_repo: Optional[ProteinRepository] = None
        self._reaction_repo: Optional[ReactionRepository] = None
        self._enzyme_reaction_repo: Optional[EnzymeReactionRepository] = None

    def encode_proteins(
        self,
        sequences: Union[str, List[str]],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Encode protein sequence(s) to embeddings.

        Args:
            sequences: Single sequence or list of sequences
            batch_size: Batch size for processing (None = all at once)

        Returns:
            Numpy array of embeddings, shape (N, D)
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        with torch.no_grad():
            embeddings = self.model.encode_proteins(
                sequences,
                device=self.device,
                batch_size=batch_size
            )

        return embeddings.cpu().numpy()

    def encode_reactions(
        self,
        reactions: Union[str, List[str]],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Encode chemical reaction(s) to embeddings.

        Args:
            reactions: Single SMILES or list of SMILES
            batch_size: Batch size for processing

        Returns:
            Numpy array of embeddings, shape (N, D)
        """
        if isinstance(reactions, str):
            reactions = [reactions]

        with torch.no_grad():
            embeddings = self.model.encode_reactions(
                reactions,
                device=self.device,
                batch_size=batch_size
            )

        return embeddings.cpu().numpy()

    def compute_similarity(
        self,
        proteins: Union[str, List[str]],
        reactions: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Compute protein-reaction similarity matrix.

        Args:
            proteins: Protein sequence(s)
            reactions: Reaction SMILES string(s)

        Returns:
            Similarity matrix (n_proteins, n_reactions)
            Higher values = more similar
        """
        if isinstance(proteins, str):
            proteins = [proteins]
        if isinstance(reactions, str):
            reactions = [reactions]

        with torch.no_grad():
            similarity = self.model.compute_similarity(
                proteins,
                reactions,
                device=self.device
            )

        return similarity.cpu().numpy()

    def find_best_reactions_for_protein(
        self,
        protein: str,
        reactions: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find top reactions for a given protein.

        Args:
            protein: Protein sequence
            reactions: List of reaction SMILES
            top_k: Number of top matches to return

        Returns:
            List of dicts with keys: 'reaction', 'score', 'rank'
        """
        similarities = self.compute_similarity(protein, reactions).flatten()

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                'reaction': reactions[idx],
                'score': float(similarities[idx]),
                'rank': rank
            })

        return results

    def find_best_proteins_for_reaction(
        self,
        reaction: str,
        proteins: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find top proteins for a given reaction.

        Args:
            reaction: Reaction SMILES
            proteins: List of protein sequences
            top_k: Number of top matches to return

        Returns:
            List of dicts with keys: 'protein', 'score', 'rank'
        """
        similarities = self.compute_similarity(proteins, reaction).flatten()

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                'protein': proteins[idx],
                'score': float(similarities[idx]),
                'rank': rank
            })

        return results

    def load_pretrained(self, model_path: str):
        """
        Load pretrained model weights.

        Args:
            model_path: Path to model directory or checkpoint
        """
        if Path(model_path).is_dir():
            # Load from save_pretrained format
            self.model = CLIPZymeModel.from_pretrained(
                model_path,
                self.model.protein_encoder,
                self.model.reaction_encoder
            )
        else:
            # Load state dict
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model = self.model.to(self.device)
        self.model.eval()

    def save_pretrained(self, save_path: str):
        """
        Save model weights and configuration.

        Args:
            save_path: Path to save directory
        """
        self.model.save_pretrained(save_path)
        self.config.to_yaml(str(Path(save_path) / 'config.yaml'))

    # Data repository methods

    def load_proteins_from_csv(self, csv_path: Optional[str] = None, **filters):
        """
        Load proteins from CSV file.

        Args:
            csv_path: Path to CSV (uses default if None)
            **filters: Filters to apply (max_length, min_length, etc.)

        Returns:
            List of Protein objects
        """
        if csv_path is None:
            csv_path = DataConfig.PROTEINS_CSV

        if self._protein_repo is None or self._protein_repo.data_path != Path(csv_path):
            self._protein_repo = ProteinRepository(csv_path)

        return self._protein_repo.load_all(**filters)

    def load_reactions_from_csv(self, csv_path: Optional[str] = None, **filters):
        """
        Load reactions from CSV file.

        Args:
            csv_path: Path to CSV (uses default if None)
            **filters: Filters to apply

        Returns:
            List of Reaction objects
        """
        if csv_path is None:
            csv_path = DataConfig.REACTIONS_EXTENDED_CSV

        if self._reaction_repo is None or self._reaction_repo.data_path != Path(csv_path):
            self._reaction_repo = ReactionRepository(csv_path)

        return self._reaction_repo.load_all(**filters)

    def load_enzyme_reactions_from_csv(self, csv_path: Optional[str] = None, **filters):
        """
        Load enzyme-reaction pairs from CSV.

        Args:
            csv_path: Path to CSV (uses default if None)
            **filters: Filters to apply

        Returns:
            List of EnzymeReactionPair objects
        """
        if csv_path is None:
            csv_path = DataConfig.ENZYME_REACTIONS_CSV

        if self._enzyme_reaction_repo is None or self._enzyme_reaction_repo.data_path != Path(csv_path):
            self._enzyme_reaction_repo = EnzymeReactionRepository(csv_path)

        return self._enzyme_reaction_repo.load_all(**filters)

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.get_embedding_dim()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CLIPZyme(\n"
            f"  protein_encoder={type(self.model.protein_encoder).__name__},\n"
            f"  reaction_encoder={type(self.model.reaction_encoder).__name__},\n"
            f"  embedding_dim={self.get_embedding_dim()},\n"
            f"  device={self.device}\n"
            f")"
        )


__all__ = ['CLIPZyme']
