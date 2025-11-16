"""
High-level predictor API for CLIPZyme.

Provides simple interface for inference without workflow complexity.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import torch
import logging

from models.clipzyme import CLIPZymeModel
from checkpoints.loader import CheckpointLoader
from screening.screening_set import ScreeningSet


logger = logging.getLogger(__name__)


@dataclass
class ScreeningResult:
    """Result of screening a single reaction."""

    reaction_smiles: str
    top_proteins: List[str]
    scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'reaction_smiles': self.reaction_smiles,
            'top_proteins': self.top_proteins,
            'scores': self.scores,
            'metadata': self.metadata
        }


@dataclass
class PredictorConfig:
    """Configuration for predictor."""

    device: str = 'cuda'
    batch_size: int = 32
    top_k: int = 100
    cache_embeddings: bool = True


class CLIPZymePredictor:
    """
    High-level predictor for CLIPZyme.

    Provides simple API for screening reactions against proteins.

    Example:
        >>> predictor = CLIPZymePredictor.from_checkpoint('path/to/checkpoint.pt')
        >>> predictor.load_screening_set('path/to/screening_set.pkl')
        >>> result = predictor.screen('[C:1]=[O:2]>>[C:1][O:2]')
        >>> print(result.top_proteins)
    """

    def __init__(
        self,
        model: CLIPZymeModel,
        config: Optional[PredictorConfig] = None
    ):
        """
        Initialize predictor.

        Args:
            model: CLIPZyme model
            config: Predictor configuration
        """
        self.model = model
        self.config = config or PredictorConfig()

        self.model.to(self.config.device)
        self.model.eval()

        self.screening_set: Optional[ScreeningSet] = None
        self._reaction_embeddings_cache: Dict[str, torch.Tensor] = {}

        logger.info("Predictor initialized")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        config: Optional[PredictorConfig] = None,
        device: Optional[str] = None
    ) -> 'CLIPZymePredictor':
        """
        Create predictor from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            config: Predictor configuration
            device: Device to load model on

        Returns:
            CLIPZymePredictor instance
        """
        if config is None:
            config = PredictorConfig()

        if device is not None:
            config.device = device

        # Load checkpoint
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        loader = CheckpointLoader(device=config.device)
        model = loader.load(checkpoint_path)

        return cls(model, config)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_name: str = 'clipzyme_official_v1',
        config: Optional[PredictorConfig] = None,
        device: Optional[str] = None
    ) -> 'CLIPZymePredictor':
        """
        Create predictor with pretrained checkpoint.

        Downloads checkpoint from Zenodo if needed.

        Args:
            checkpoint_name: Name of pretrained checkpoint
            config: Predictor configuration
            device: Device to load model on

        Returns:
            CLIPZymePredictor instance
        """
        from checkpoints.downloader import CheckpointDownloader

        # Download checkpoint
        downloader = CheckpointDownloader()
        checkpoint_path = downloader.download(checkpoint_name)

        return cls.from_checkpoint(checkpoint_path, config, device)

    def load_screening_set(self, screening_set_path: Union[str, Path, ScreeningSet]):
        """
        Load screening set for virtual screening.

        Args:
            screening_set_path: Path to screening set file or ScreeningSet instance
        """
        if isinstance(screening_set_path, ScreeningSet):
            self.screening_set = screening_set_path
        else:
            logger.info(f"Loading screening set: {screening_set_path}")
            self.screening_set = ScreeningSet.load(screening_set_path)

        # Move to device
        self.screening_set = self.screening_set.to(self.config.device)

        logger.info(f"Loaded {len(self.screening_set)} proteins")

    def screen(
        self,
        reaction_smiles: str,
        top_k: Optional[int] = None
    ) -> ScreeningResult:
        """
        Screen a reaction against proteins.

        Args:
            reaction_smiles: Reaction SMILES string
            top_k: Number of top proteins to return

        Returns:
            ScreeningResult

        Raises:
            ValueError: If screening set not loaded
        """
        if self.screening_set is None:
            raise ValueError(
                "No screening set loaded. Call load_screening_set() first."
            )

        top_k = top_k or self.config.top_k

        # Get reaction embedding (with caching)
        if self.config.cache_embeddings and reaction_smiles in self._reaction_embeddings_cache:
            reaction_emb = self._reaction_embeddings_cache[reaction_smiles]
        else:
            with torch.no_grad():
                reaction_emb = self.model.encode_reactions([reaction_smiles])

            if self.config.cache_embeddings:
                self._reaction_embeddings_cache[reaction_smiles] = reaction_emb

        # Compute similarities
        protein_embs = self.screening_set.embeddings
        similarities = (reaction_emb @ protein_embs.T).squeeze(0)

        # Get top-k
        top_k_scores, top_k_indices = torch.topk(similarities, k=min(top_k, len(self.screening_set)))

        # Get protein names
        top_proteins = [
            self.screening_set.protein_names[idx]
            for idx in top_k_indices.cpu().tolist()
        ]

        # Convert scores to list
        scores = top_k_scores.cpu().tolist()

        return ScreeningResult(
            reaction_smiles=reaction_smiles,
            top_proteins=top_proteins,
            scores=scores,
            metadata={
                'screening_set_size': len(self.screening_set),
                'top_k': top_k
            }
        )

    def screen_batch(
        self,
        reaction_smiles_list: List[str],
        top_k: Optional[int] = None
    ) -> List[ScreeningResult]:
        """
        Screen multiple reactions.

        Args:
            reaction_smiles_list: List of reaction SMILES
            top_k: Number of top proteins per reaction

        Returns:
            List of ScreeningResults
        """
        results = []

        for reaction_smiles in reaction_smiles_list:
            result = self.screen(reaction_smiles, top_k)
            results.append(result)

        return results

    def encode_protein(self, sequence: str) -> torch.Tensor:
        """
        Encode a protein sequence.

        Args:
            sequence: Protein sequence

        Returns:
            Protein embedding tensor
        """
        with torch.no_grad():
            embedding = self.model.encode_proteins([sequence])

        return embedding

    def encode_proteins(self, sequences: List[str]) -> torch.Tensor:
        """
        Encode multiple protein sequences.

        Args:
            sequences: List of protein sequences

        Returns:
            Protein embeddings tensor [batch_size, embedding_dim]
        """
        with torch.no_grad():
            embeddings = self.model.encode_proteins(sequences)

        return embeddings

    def encode_reaction(self, reaction_smiles: str) -> torch.Tensor:
        """
        Encode a reaction.

        Args:
            reaction_smiles: Reaction SMILES

        Returns:
            Reaction embedding tensor
        """
        with torch.no_grad():
            embedding = self.model.encode_reactions([reaction_smiles])

        return embedding

    def encode_reactions(self, reaction_smiles_list: List[str]) -> torch.Tensor:
        """
        Encode multiple reactions.

        Args:
            reaction_smiles_list: List of reaction SMILES

        Returns:
            Reaction embeddings tensor [batch_size, embedding_dim]
        """
        with torch.no_grad():
            embeddings = self.model.encode_reactions(reaction_smiles_list)

        return embeddings

    def compute_similarity(
        self,
        protein_sequence: str,
        reaction_smiles: str
    ) -> float:
        """
        Compute similarity between protein and reaction.

        Args:
            protein_sequence: Protein sequence
            reaction_smiles: Reaction SMILES

        Returns:
            Similarity score (cosine similarity)
        """
        protein_emb = self.encode_protein(protein_sequence)
        reaction_emb = self.encode_reaction(reaction_smiles)

        similarity = (protein_emb @ reaction_emb.T).item()

        return similarity

    def clear_cache(self):
        """Clear reaction embeddings cache."""
        self._reaction_embeddings_cache.clear()
        logger.info("Cache cleared")

    def to(self, device: str) -> 'CLIPZymePredictor':
        """
        Move model to device.

        Args:
            device: Target device

        Returns:
            Self
        """
        self.model.to(device)
        self.config.device = device

        if self.screening_set is not None:
            self.screening_set = self.screening_set.to(device)

        return self


# Export
__all__ = [
    'CLIPZymePredictor',
    'PredictorConfig',
    'ScreeningResult',
]
