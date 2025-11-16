"""
Screening Set Management

Handles large collections of pre-embedded proteins for virtual screening.
Supports loading from pickle files, CSV files, or building from scratch.

Format compatible with CLIPZyme official:
- screening_set.p: Dict[str, torch.Tensor] - protein_id -> embedding
- uniprot2sequence.p: Dict[str, str] - protein_id -> amino acid sequence
"""

import torch
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProteinEntry:
    """Single protein entry in the screening set."""
    protein_id: str
    sequence: str
    embedding: Optional[torch.Tensor] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ProteinDatabase:
    """
    Database of proteins with sequences and optional metadata.

    Supports:
    - Loading from CSV (protein_id, sequence, optional columns)
    - Loading from pickle (uniprot2sequence.p format)
    - Querying and filtering
    """

    def __init__(self):
        self.proteins: Dict[str, ProteinEntry] = {}

    def load_from_csv(
        self,
        csv_path: Union[str, Path],
        id_column: str = "protein_id",
        sequence_column: str = "sequence",
        metadata_columns: Optional[List[str]] = None
    ) -> 'ProteinDatabase':
        """
        Load proteins from CSV file.

        Args:
            csv_path: Path to CSV file
            id_column: Column name for protein IDs
            sequence_column: Column name for sequences
            metadata_columns: Additional columns to store as metadata

        Returns:
            Self for chaining
        """
        df = pd.read_csv(csv_path)

        if id_column not in df.columns:
            raise ValueError(f"Column '{id_column}' not found in CSV")
        if sequence_column not in df.columns:
            raise ValueError(f"Column '{sequence_column}' not found in CSV")

        metadata_columns = metadata_columns or []

        for idx, row in df.iterrows():
            protein_id = str(row[id_column])
            sequence = str(row[sequence_column])

            metadata = {col: row[col] for col in metadata_columns if col in df.columns}

            self.proteins[protein_id] = ProteinEntry(
                protein_id=protein_id,
                sequence=sequence,
                metadata=metadata
            )

        logger.info(f"Loaded {len(self.proteins)} proteins from {csv_path}")
        return self

    def load_from_pickle(
        self,
        pickle_path: Union[str, Path]
    ) -> 'ProteinDatabase':
        """
        Load proteins from pickle file (uniprot2sequence.p format).

        Args:
            pickle_path: Path to pickle file with Dict[str, str] mapping

        Returns:
            Self for chaining
        """
        with open(pickle_path, 'rb') as f:
            protein_dict = pickle.load(f)

        for protein_id, sequence in protein_dict.items():
            self.proteins[protein_id] = ProteinEntry(
                protein_id=protein_id,
                sequence=sequence
            )

        logger.info(f"Loaded {len(self.proteins)} proteins from {pickle_path}")
        return self

    def save_to_pickle(self, pickle_path: Union[str, Path]):
        """Save protein sequences to pickle file."""
        protein_dict = {pid: entry.sequence for pid, entry in self.proteins.items()}
        with open(pickle_path, 'wb') as f:
            pickle.dump(protein_dict, f)
        logger.info(f"Saved {len(self.proteins)} proteins to {pickle_path}")

    def get_sequence(self, protein_id: str) -> Optional[str]:
        """Get sequence for a protein ID."""
        entry = self.proteins.get(protein_id)
        return entry.sequence if entry else None

    def get_sequences(self, protein_ids: List[str]) -> List[str]:
        """Get sequences for multiple protein IDs."""
        sequences = []
        for pid in protein_ids:
            seq = self.get_sequence(pid)
            if seq:
                sequences.append(seq)
        return sequences

    def __len__(self) -> int:
        return len(self.proteins)

    def __contains__(self, protein_id: str) -> bool:
        return protein_id in self.proteins

    def __getitem__(self, protein_id: str) -> ProteinEntry:
        return self.proteins[protein_id]


class ScreeningSet:
    """
    Collection of pre-embedded proteins for virtual screening.

    This is the main class for managing screening sets. It can:
    - Load pre-computed embeddings from pickle files
    - Store embeddings in memory or on disk
    - Query embeddings efficiently
    - Compute similarity with reaction embeddings

    Compatible with CLIPZyme official format:
    - screening_set.p: Dict[str, Tensor] with protein_id -> embedding (512D)
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        device: str = "cpu"
    ):
        """
        Initialize screening set.

        Args:
            embedding_dim: Dimension of protein embeddings
            device: Device to store embeddings on
        """
        self.embedding_dim = embedding_dim
        self.device = device
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.protein_ids: List[str] = []

    def load_from_pickle(
        self,
        pickle_path: Union[str, Path],
        device: Optional[str] = None
    ) -> 'ScreeningSet':
        """
        Load pre-computed embeddings from pickle file.

        Args:
            pickle_path: Path to screening_set.p file
            device: Device to load embeddings to (default: self.device)

        Returns:
            Self for chaining
        """
        device = device or self.device

        logger.info(f"Loading screening set from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            embeddings_dict = pickle.load(f)

        # Load embeddings
        for protein_id, embedding in embeddings_dict.items():
            if isinstance(embedding, torch.Tensor):
                self.embeddings[protein_id] = embedding.to(device)
            else:
                # Convert numpy array to tensor
                self.embeddings[protein_id] = torch.tensor(embedding, device=device)

        self.protein_ids = list(self.embeddings.keys())

        # Verify dimensions
        first_emb = next(iter(self.embeddings.values()))
        if first_emb.shape[-1] != self.embedding_dim:
            logger.warning(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {first_emb.shape[-1]}"
            )
            self.embedding_dim = first_emb.shape[-1]

        logger.info(
            f"Loaded {len(self.embeddings)} protein embeddings "
            f"({self.embedding_dim}D) on {device}"
        )
        return self

    def save_to_pickle(self, pickle_path: Union[str, Path]):
        """
        Save embeddings to pickle file.

        Args:
            pickle_path: Output path for screening_set.p
        """
        # Move to CPU for saving
        embeddings_dict = {
            pid: emb.cpu() for pid, emb in self.embeddings.items()
        }

        with open(pickle_path, 'wb') as f:
            pickle.dump(embeddings_dict, f)

        logger.info(f"Saved {len(embeddings_dict)} embeddings to {pickle_path}")

    def add_embedding(
        self,
        protein_id: str,
        embedding: torch.Tensor
    ):
        """
        Add a single protein embedding.

        Args:
            protein_id: Protein identifier
            embedding: Embedding tensor (embedding_dim,)
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        if embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embedding.shape[-1]}"
            )

        self.embeddings[protein_id] = embedding.squeeze(0).to(self.device)
        if protein_id not in self.protein_ids:
            self.protein_ids.append(protein_id)

    def add_embeddings_batch(
        self,
        protein_ids: List[str],
        embeddings: torch.Tensor
    ):
        """
        Add multiple protein embeddings at once.

        Args:
            protein_ids: List of protein identifiers
            embeddings: Tensor of embeddings (batch_size, embedding_dim)
        """
        if len(protein_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(protein_ids)} IDs but {embeddings.shape[0]} embeddings"
            )

        for pid, emb in zip(protein_ids, embeddings):
            self.add_embedding(pid, emb)

    def get_embedding(self, protein_id: str) -> Optional[torch.Tensor]:
        """Get embedding for a single protein."""
        return self.embeddings.get(protein_id)

    def get_embeddings_tensor(self) -> Tuple[torch.Tensor, List[str]]:
        """
        Get all embeddings as a single tensor.

        Returns:
            embeddings: Tensor (num_proteins, embedding_dim)
            protein_ids: List of protein IDs in same order
        """
        if not self.protein_ids:
            return torch.empty(0, self.embedding_dim, device=self.device), []

        embeddings = torch.stack([
            self.embeddings[pid] for pid in self.protein_ids
        ])

        return embeddings, self.protein_ids

    def compute_similarity(
        self,
        reaction_embedding: torch.Tensor,
        top_k: Optional[int] = None,
        return_scores: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, List[str]]]:
        """
        Compute similarity between a reaction and all proteins.

        Args:
            reaction_embedding: Reaction embedding (embedding_dim,) or (1, embedding_dim)
            top_k: If provided, return only top-k matches
            return_scores: If True, return (scores, indices, protein_ids)

        Returns:
            If return_scores=False: similarity scores (num_proteins,)
            If return_scores=True: (scores, indices, protein_ids) for top-k matches
        """
        # Get all embeddings as tensor
        protein_embeddings, protein_ids = self.get_embeddings_tensor()

        if protein_embeddings.shape[0] == 0:
            if return_scores:
                return torch.tensor([]), torch.tensor([]), []
            return torch.tensor([])

        # Ensure reaction embedding is 1D
        if reaction_embedding.dim() == 2:
            reaction_embedding = reaction_embedding.squeeze(0)

        # Move to same device
        reaction_embedding = reaction_embedding.to(self.device)

        # Compute cosine similarity (embeddings are L2-normalized)
        similarities = protein_embeddings @ reaction_embedding

        if not return_scores:
            return similarities

        # Get top-k
        if top_k is None:
            top_k = len(protein_ids)
        else:
            top_k = min(top_k, len(protein_ids))

        scores, indices = torch.topk(similarities, k=top_k, largest=True)
        top_protein_ids = [protein_ids[idx] for idx in indices.cpu().tolist()]

        return scores, indices, top_protein_ids

    def filter_by_ids(self, protein_ids: List[str]) -> 'ScreeningSet':
        """
        Create a new ScreeningSet with only specified protein IDs.

        Args:
            protein_ids: List of protein IDs to keep

        Returns:
            New ScreeningSet with filtered proteins
        """
        filtered = ScreeningSet(
            embedding_dim=self.embedding_dim,
            device=self.device
        )

        for pid in protein_ids:
            if pid in self.embeddings:
                filtered.add_embedding(pid, self.embeddings[pid])

        return filtered

    def __len__(self) -> int:
        return len(self.embeddings)

    def __contains__(self, protein_id: str) -> bool:
        return protein_id in self.embeddings

    def __repr__(self) -> str:
        return (
            f"ScreeningSet(num_proteins={len(self)}, "
            f"embedding_dim={self.embedding_dim}, device={self.device})"
        )


def build_screening_set_from_model(
    model,
    protein_database: ProteinDatabase,
    batch_size: int = 32,
    device: str = "cuda",
    show_progress: bool = True
) -> ScreeningSet:
    """
    Build a screening set by encoding proteins with a model.

    Args:
        model: CLIPZyme model or protein encoder
        protein_database: Database with protein sequences
        batch_size: Batch size for encoding
        device: Device for encoding
        show_progress: Show progress bar

    Returns:
        ScreeningSet with all encoded proteins
    """
    from tqdm import tqdm

    model = model.to(device)
    model.eval()

    # Get embedding dimension
    embedding_dim = model.get_embedding_dim() if hasattr(model, 'get_embedding_dim') else 512

    screening_set = ScreeningSet(embedding_dim=embedding_dim, device=device)

    protein_ids = list(protein_database.proteins.keys())
    sequences = [protein_database.proteins[pid].sequence for pid in protein_ids]

    # Encode in batches
    num_batches = (len(sequences) + batch_size - 1) // batch_size

    iterator = range(num_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Encoding proteins")

    with torch.no_grad():
        for i in iterator:
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(sequences))

            batch_ids = protein_ids[start_idx:end_idx]
            batch_seqs = sequences[start_idx:end_idx]

            # Encode batch
            if hasattr(model, 'encode_proteins'):
                embeddings = model.encode_proteins(batch_seqs, device=device)
            else:
                # Assume model is protein encoder
                embeddings = model.encode(batch_seqs, device=device)

            # Add to screening set
            screening_set.add_embeddings_batch(batch_ids, embeddings)

    logger.info(f"Built screening set with {len(screening_set)} proteins")
    return screening_set


__all__ = [
    'ProteinEntry',
    'ProteinDatabase',
    'ScreeningSet',
    'build_screening_set_from_model',
]
