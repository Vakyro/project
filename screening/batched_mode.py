"""
Batched Screening Mode

High-throughput screening with multi-GPU support.
Optimized for screening thousands of reactions efficiently.

Usage:
    screener = BatchedScreener(model, screening_set)
    results = screener.screen_reactions(reactions_list, batch_size=64)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json

from screening.screening_set import ScreeningSet, ProteinDatabase
from screening.ranking import (
    ScreeningResult,
    rank_proteins_for_reaction,
    evaluate_screening,
    batch_evaluate_screening
)

logger = logging.getLogger(__name__)


@dataclass
class BatchedScreeningConfig:
    """Configuration for batched screening."""
    batch_size: int = 64
    num_workers: int = 4
    top_k: int = 100
    devices: List[str] = None  # List of device strings, e.g., ['cuda:0', 'cuda:1']
    save_embeddings: bool = False
    save_scores: bool = True
    output_dir: Optional[str] = None
    evaluate: bool = False

    def __post_init__(self):
        if self.devices is None:
            # Default to single GPU if available
            if torch.cuda.is_available():
                self.devices = ['cuda:0']
            else:
                self.devices = ['cpu']


class ReactionScreeningDataset(Dataset):
    """Dataset for batched screening."""

    def __init__(
        self,
        reaction_smiles_list: List[str],
        reaction_ids: Optional[List[str]] = None,
        true_positives_list: Optional[List[List[str]]] = None
    ):
        """
        Initialize dataset.

        Args:
            reaction_smiles_list: List of reaction SMILES strings
            reaction_ids: Optional list of reaction identifiers
            true_positives_list: Optional list of true positive protein IDs per reaction
        """
        self.reaction_smiles_list = reaction_smiles_list
        self.reaction_ids = reaction_ids or [
            f"reaction_{i}" for i in range(len(reaction_smiles_list))
        ]
        self.true_positives_list = true_positives_list or [None] * len(reaction_smiles_list)

    def __len__(self) -> int:
        return len(self.reaction_smiles_list)

    def __getitem__(self, idx: int) -> Dict:
        return {
            'reaction_smiles': self.reaction_smiles_list[idx],
            'reaction_id': self.reaction_ids[idx],
            'true_positives': self.true_positives_list[idx],
            'idx': idx
        }


class BatchedScreener:
    """
    Batched screening mode for high-throughput virtual screening.

    Features:
    - Batch processing for efficiency
    - Multi-GPU support via DataParallel
    - Progress tracking
    - Results saving (embeddings, scores, rankings)
    - Memory-efficient streaming mode

    Suitable for:
    - Large-scale screening (>1000 reactions)
    - Production deployments
    - Benchmark evaluations
    """

    def __init__(
        self,
        model,
        screening_set: ScreeningSet,
        protein_database: Optional[ProteinDatabase] = None,
        config: Optional[BatchedScreeningConfig] = None
    ):
        """
        Initialize batched screener.

        Args:
            model: CLIPZyme model or tuple of (protein_encoder, reaction_encoder)
            screening_set: ScreeningSet with pre-embedded proteins
            protein_database: Optional database with protein sequences/metadata
            config: Screening configuration
        """
        self.config = config or BatchedScreeningConfig()

        # Handle model types
        if isinstance(model, tuple):
            self.protein_encoder, self.reaction_encoder = model
            self.model = None
        else:
            self.model = model
            self.protein_encoder = model.protein_encoder if hasattr(model, 'protein_encoder') else None
            self.reaction_encoder = model.reaction_encoder if hasattr(model, 'reaction_encoder') else None

        self.screening_set = screening_set
        self.protein_database = protein_database

        # Setup devices
        self.primary_device = self.config.devices[0]
        self.use_multi_gpu = len(self.config.devices) > 1 and torch.cuda.is_available()

        # Move models to device(s)
        self._setup_models()

        # Create output directory if needed
        if self.config.output_dir:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized BatchedScreener with {len(screening_set)} proteins "
            f"on {self.config.devices}"
        )

    def _setup_models(self):
        """Setup models on device(s) with optional DataParallel."""
        if self.model is not None:
            self.model = self.model.to(self.primary_device)
            if self.use_multi_gpu:
                device_ids = [int(d.split(':')[1]) for d in self.config.devices if 'cuda' in d]
                self.model = nn.DataParallel(self.model, device_ids=device_ids)
            self.model.eval()

        if self.reaction_encoder is not None:
            self.reaction_encoder = self.reaction_encoder.to(self.primary_device)
            if self.use_multi_gpu:
                device_ids = [int(d.split(':')[1]) for d in self.config.devices if 'cuda' in d]
                self.reaction_encoder = nn.DataParallel(self.reaction_encoder, device_ids=device_ids)
            self.reaction_encoder.eval()

    def encode_reactions_batch(
        self,
        reaction_smiles_list: List[str]
    ) -> torch.Tensor:
        """
        Encode a batch of reactions.

        Args:
            reaction_smiles_list: List of reaction SMILES

        Returns:
            Batch of reaction embeddings (batch_size, embedding_dim)
        """
        with torch.no_grad():
            if self.model is not None:
                # Use full model
                if hasattr(self.model, 'module'):
                    # DataParallel wrapped
                    encoder = self.model.module
                else:
                    encoder = self.model

                if hasattr(encoder, 'encode_reactions'):
                    embeddings = encoder.encode_reactions(
                        reaction_smiles_list,
                        device=self.primary_device
                    )
                else:
                    raise RuntimeError("Model does not have encode_reactions method")

            elif self.reaction_encoder is not None:
                # Use reaction encoder directly
                if hasattr(self.reaction_encoder, 'module'):
                    encoder = self.reaction_encoder.module
                else:
                    encoder = self.reaction_encoder

                embeddings = encoder.encode(
                    reaction_smiles_list,
                    device=self.primary_device
                )
            else:
                raise RuntimeError("No reaction encoder available")

            return embeddings

    def screen_batch(
        self,
        batch: Dict,
        top_k: Optional[int] = None
    ) -> List[ScreeningResult]:
        """
        Screen a batch of reactions.

        Args:
            batch: Batch dictionary from DataLoader
            top_k: Number of top matches per reaction

        Returns:
            List of ScreeningResult objects
        """
        top_k = top_k or self.config.top_k

        reaction_smiles_list = batch['reaction_smiles']
        reaction_ids = batch['reaction_id']
        true_positives_list = batch['true_positives']

        # Encode reactions
        try:
            reaction_embeddings = self.encode_reactions_batch(reaction_smiles_list)
        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            raise

        # Screen each reaction
        results = []
        for i, (rxn_id, rxn_emb, true_pos) in enumerate(
            zip(reaction_ids, reaction_embeddings, true_positives_list)
        ):
            # Rank proteins
            ranked_protein_ids, scores, _ = rank_proteins_for_reaction(
                reaction_embedding=rxn_emb,
                screening_set=self.screening_set,
                top_k=top_k,
                return_all_scores=False
            )

            # Build result
            result = ScreeningResult(
                reaction_id=rxn_id,
                ranked_protein_ids=ranked_protein_ids,
                scores=scores,
                true_positives=true_pos if true_pos is not None else None
            )

            # Evaluate if requested
            if self.config.evaluate and true_pos:
                metrics = evaluate_screening(
                    ranked_protein_ids=ranked_protein_ids,
                    scores=scores,
                    true_positives=true_pos
                )
                result.metrics = metrics

            results.append(result)

        return results

    def screen_reactions(
        self,
        reaction_smiles_list: List[str],
        reaction_ids: Optional[List[str]] = None,
        true_positives_list: Optional[List[List[str]]] = None,
        show_progress: bool = True
    ) -> List[ScreeningResult]:
        """
        Screen multiple reactions in batches.

        Args:
            reaction_smiles_list: List of reaction SMILES
            reaction_ids: Optional identifiers for reactions
            true_positives_list: Optional list of true positives for each reaction
            show_progress: Show progress bar

        Returns:
            List of ScreeningResult objects
        """
        from tqdm import tqdm

        # Create dataset and dataloader
        dataset = ReactionScreeningDataset(
            reaction_smiles_list=reaction_smiles_list,
            reaction_ids=reaction_ids,
            true_positives_list=true_positives_list
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            collate_fn=self._collate_fn
        )

        all_results = []

        iterator = dataloader
        if show_progress:
            iterator = tqdm(
                dataloader,
                desc="Screening batches",
                total=len(dataloader)
            )

        for batch in iterator:
            try:
                batch_results = self.screen_batch(batch)
                all_results.extend(batch_results)

                # Save intermediate results if output_dir specified
                if self.config.output_dir:
                    self._save_batch_results(batch_results)

            except Exception as e:
                logger.error(f"Failed to screen batch: {e}")
                continue

        logger.info(f"Screened {len(all_results)} reactions")

        # Compute aggregate metrics if evaluation enabled
        if self.config.evaluate:
            aggregate_metrics = batch_evaluate_screening(all_results)
            logger.info(f"Aggregate metrics: {aggregate_metrics}")

            # Save metrics
            if self.config.output_dir:
                self._save_metrics(aggregate_metrics)

        return all_results

    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Collate function for DataLoader."""
        return {
            'reaction_smiles': [item['reaction_smiles'] for item in batch],
            'reaction_id': [item['reaction_id'] for item in batch],
            'true_positives': [item['true_positives'] for item in batch],
            'idx': [item['idx'] for item in batch]
        }

    def _save_batch_results(self, results: List[ScreeningResult]):
        """Save batch results to disk."""
        if not self.config.output_dir:
            return

        output_dir = Path(self.config.output_dir)

        for result in results:
            # Save rankings
            if self.config.save_scores:
                ranking_file = output_dir / f"{result.reaction_id}_ranking.json"
                ranking_data = {
                    'reaction_id': result.reaction_id,
                    'ranked_protein_ids': result.ranked_protein_ids,
                    'scores': result.scores.cpu().tolist(),
                    'metrics': result.metrics
                }
                with open(ranking_file, 'w') as f:
                    json.dump(ranking_data, f, indent=2)

    def _save_metrics(self, metrics: Dict):
        """Save aggregate metrics to disk."""
        if not self.config.output_dir:
            return

        metrics_file = Path(self.config.output_dir) / "aggregate_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {metrics_file}")

    def screen_from_csv(
        self,
        csv_path: str,
        reaction_column: str = "reaction_smiles",
        id_column: Optional[str] = None,
        true_positives_column: Optional[str] = None,
        show_progress: bool = True
    ) -> List[ScreeningResult]:
        """
        Screen reactions from a CSV file.

        Args:
            csv_path: Path to CSV file
            reaction_column: Column name for reaction SMILES
            id_column: Column name for reaction IDs (optional)
            true_positives_column: Column for true positive protein IDs (optional)
            show_progress: Show progress bar

        Returns:
            List of ScreeningResult objects
        """
        import pandas as pd

        df = pd.read_csv(csv_path)

        if reaction_column not in df.columns:
            raise ValueError(f"Column '{reaction_column}' not found in CSV")

        reaction_smiles_list = df[reaction_column].tolist()

        # Get IDs
        if id_column and id_column in df.columns:
            reaction_ids = df[id_column].tolist()
        else:
            reaction_ids = None

        # Get true positives
        if true_positives_column and true_positives_column in df.columns:
            # Assume format: "protein1,protein2,protein3"
            true_positives_list = [
                str(tp).split(',') if pd.notna(tp) else None
                for tp in df[true_positives_column]
            ]
        else:
            true_positives_list = None

        return self.screen_reactions(
            reaction_smiles_list=reaction_smiles_list,
            reaction_ids=reaction_ids,
            true_positives_list=true_positives_list,
            show_progress=show_progress
        )

    def __repr__(self) -> str:
        return (
            f"BatchedScreener("
            f"num_proteins={len(self.screening_set)}, "
            f"batch_size={self.config.batch_size}, "
            f"devices={self.config.devices})"
        )


__all__ = [
    'BatchedScreeningConfig',
    'ReactionScreeningDataset',
    'BatchedScreener',
]
