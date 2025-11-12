"""
Dataset and batching utilities for reaction graphs.
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
    Dataset for reaction SMILES.

    Args:
        rxn_smiles_list: List of reaction SMILES strings (with atom mapping)
        preprocess: If True, build graphs during initialization; otherwise build on-the-fly
    """

    def __init__(self, rxn_smiles_list: List[str], preprocess: bool = True):
        self.rxn_smiles_list = rxn_smiles_list
        self.preprocess = preprocess
        self.graphs = []

        if preprocess:
            print(f"Preprocessing {len(rxn_smiles_list)} reactions...")
            for i, rxn in enumerate(rxn_smiles_list):
                try:
                    reacts, prods = parse_reaction_smiles(rxn)
                    graph = build_transition_graph(reacts, prods)
                    self.graphs.append(graph)
                except Exception as e:
                    print(f"Warning: Failed to process reaction {i}: {e}")
                    self.graphs.append(None)

    def __len__(self):
        return len(self.rxn_smiles_list)

    def __getitem__(self, idx: int) -> Data:
        if self.preprocess:
            graph = self.graphs[idx]
            if graph is None:
                raise ValueError(f"Graph at index {idx} failed preprocessing")
            return graph
        else:
            rxn = self.rxn_smiles_list[idx]
            reacts, prods = parse_reaction_smiles(rxn)
            return build_transition_graph(reacts, prods)


def create_dataloader(rxn_smiles_list: List[str], batch_size: int = 32,
                      shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Create a DataLoader for reaction graphs.

    Args:
        rxn_smiles_list: List of reaction SMILES
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        PyTorch Geometric DataLoader
    """
    dataset = ReactionDataset(rxn_smiles_list, preprocess=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
