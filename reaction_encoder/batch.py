"""
Dataset and batching utilities for reaction graphs.

This module provides PyTorch Dataset and DataLoader utilities for processing
chemical reactions represented as graph structures.
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List, Tuple

from .chem import parse_reaction_smiles
from .builder import build_transition_graph


class ReactionDataset(Dataset):
    """
    PyTorch Dataset for chemical reactions with transition state graphs.

    This dataset handles atom-mapped reaction SMILES strings and converts them
    into PyTorch Geometric graph objects suitable for GNN processing.

    Features:
        - Supports preprocessing (build all graphs at init) or lazy loading
        - Handles parsing errors gracefully by storing None for failed reactions
        - Compatible with PyTorch Geometric DataLoader for batching

    Args:
        reaction_smiles_list: List of atom-mapped reaction SMILES strings
                             Example: ["[C:1]=[O:2]>>[C:1]-[O:2]", ...]
        preprocess: If True, build all graphs during initialization.
                   If False, build graphs on-the-fly during __getitem__.
                   Preprocessing is faster for training but uses more memory.

    Attributes:
        reaction_smiles_list: Original SMILES strings
        preprocess: Whether graphs are preprocessed
        graphs: List of preprocessed graphs (empty if preprocess=False)
    """

    def __init__(self, reaction_smiles_list: List[str], preprocess: bool = True):
        self.reaction_smiles_list = reaction_smiles_list
        self.preprocess = preprocess
        self.graphs = []

        # Build all graphs upfront if preprocessing is enabled
        if preprocess:
            for reaction_idx, reaction_smiles in enumerate(reaction_smiles_list):
                try:
                    # Parse reaction SMILES into reactant and product molecules
                    reactant_molecules, product_molecules = parse_reaction_smiles(reaction_smiles)

                    # Build transition state graph representation
                    transition_graph = build_transition_graph(reactant_molecules, product_molecules)

                    self.graphs.append(transition_graph)
                except Exception:
                    # Store None for reactions that fail to parse
                    # This allows the dataset to continue loading despite individual failures
                    self.graphs.append(None)

    def __len__(self) -> int:
        """Return the total number of reactions in the dataset."""
        return len(self.reaction_smiles_list)

    def __getitem__(self, index: int) -> Data:
        """
        Get a single reaction graph by index.

        Args:
            index: Index of the reaction to retrieve

        Returns:
            PyTorch Geometric Data object representing the reaction's transition state

        Raises:
            ValueError: If preprocessed graph at index is None (failed to parse)
        """
        if self.preprocess:
            # Return preprocessed graph
            graph = self.graphs[index]
            if graph is None:
                raise ValueError(f"Graph at index {index} failed during preprocessing")
            return graph
        else:
            # Build graph on-the-fly (lazy loading)
            reaction_smiles = self.reaction_smiles_list[index]
            reactant_molecules, product_molecules = parse_reaction_smiles(reaction_smiles)
            return build_transition_graph(reactant_molecules, product_molecules)


def create_dataloader(
    reaction_smiles_list: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for batching reaction transition state graphs.

    This convenience function creates a preprocessed ReactionDataset and wraps it
    in a PyTorch Geometric DataLoader for efficient batched training.

    Args:
        reaction_smiles_list: List of atom-mapped reaction SMILES strings
        batch_size: Number of reactions per batch (default: 32)
        shuffle: Whether to shuffle the data each epoch (default: True).
                Recommended for training, disable for validation/test.
        num_workers: Number of parallel worker processes for data loading (default: 0).
                    Set to 0 for single-process loading, >0 for multiprocessing.

    Returns:
        PyTorch Geometric DataLoader configured for reaction graph batching

    Example:
        >>> reactions = ["[C:1]=[O:2]>>[C:1]-[O:2]", "[N:1]=[N:2]>>[N:1]-[N:2]"]
        >>> loader = create_dataloader(reactions, batch_size=2, shuffle=True)
        >>> for batch in loader:
        ...     # batch.x: node features, batch.edge_index: edges, etc.
        ...     pass
    """
    # Create dataset with preprocessing enabled for faster iteration
    dataset = ReactionDataset(reaction_smiles_list, preprocess=True)

    # Wrap in DataLoader for batching
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
