#!/usr/bin/env python3
"""
Build Screening Set

Encodes a collection of proteins and creates a screening set file (screening_set.p).

Usage:
    python scripts/build_screening_set.py --config configs/build_screening_set.yaml
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from screening.screening_set import (
    ScreeningSet,
    ProteinDatabase,
    build_screening_set_from_model
)
from models.clipzyme import CLIPZymeModel
from models.builder import CLIPZymeBuilder
from config.config import CLIPZymeConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config: dict):
    """Load CLIPZyme model."""
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    if config.get('checkpoint_path'):
        logger.info(f"Loading model from checkpoint: {config['checkpoint_path']}")
        model = torch.load(config['checkpoint_path'], map_location=device)
    else:
        logger.info(f"Building model from config: {config['config_path']}")
        clipzyme_config = CLIPZymeConfig.from_yaml(config['config_path'])
        builder = CLIPZymeBuilder()
        model = builder.with_config(clipzyme_config).build()

    model = model.to(device)
    model.eval()
    return model


def load_protein_database(config: dict) -> ProteinDatabase:
    """Load protein database."""
    protein_db = ProteinDatabase()

    if config.get('csv_path'):
        logger.info(f"Loading proteins from CSV: {config['csv_path']}")

        # Apply filters
        import pandas as pd
        df = pd.read_csv(config['csv_path'])

        # Filter by length
        if config.get('filter_max_length'):
            seq_col = config.get('sequence_column', 'sequence')
            df = df[df[seq_col].str.len() <= config['filter_max_length']]
            logger.info(f"Filtered to max length {config['filter_max_length']}: {len(df)} proteins")

        if config.get('filter_min_length'):
            seq_col = config.get('sequence_column', 'sequence')
            df = df[df[seq_col].str.len() >= config['filter_min_length']]
            logger.info(f"Filtered to min length {config['filter_min_length']}: {len(df)} proteins")

        # Save filtered CSV temporarily
        temp_csv = Path(config['csv_path']).parent / "temp_filtered.csv"
        df.to_csv(temp_csv, index=False)

        protein_db.load_from_csv(
            csv_path=temp_csv,
            id_column=config.get('id_column', 'protein_id'),
            sequence_column=config.get('sequence_column', 'sequence')
        )

        # Clean up
        temp_csv.unlink()

    elif config.get('pickle_path'):
        logger.info(f"Loading proteins from pickle: {config['pickle_path']}")
        protein_db.load_from_pickle(config['pickle_path'])

    else:
        raise ValueError("Must provide csv_path or pickle_path")

    return protein_db


def main():
    parser = argparse.ArgumentParser(description="Build CLIPZyme Screening Set")

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Override output path for screening set'
    )

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Load model
    logger.info("Loading model...")
    model = load_model(config['model'])

    # Load protein database
    logger.info("Loading protein database...")
    protein_db = load_protein_database(config['proteins'])

    logger.info(f"Loaded {len(protein_db)} proteins")

    # Build screening set
    logger.info("Encoding proteins...")
    screening_set = build_screening_set_from_model(
        model=model,
        protein_database=protein_db,
        batch_size=config['encoding'].get('batch_size', 32),
        device=config['model']['device'],
        show_progress=config['encoding'].get('show_progress', True)
    )

    logger.info(f"Built screening set with {len(screening_set)} proteins")

    # Save screening set
    output_path = args.output or config['output']['screening_set_path']
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving screening set to {output_path}")
    screening_set.save_to_pickle(output_path)

    # Save protein database if requested
    if config['output'].get('protein_db_path'):
        db_path = Path(config['output']['protein_db_path'])
        db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving protein database to {db_path}")
        protein_db.save_to_pickle(db_path)

    logger.info("Done!")

    # Print statistics
    logger.info(f"\n=== Statistics ===")
    logger.info(f"Total proteins: {len(screening_set)}")
    logger.info(f"Embedding dimension: {screening_set.embedding_dim}")
    logger.info(f"Device: {screening_set.device}")

    # Estimate file sizes
    import os
    if output_path.exists():
        size_mb = os.path.getsize(output_path) / (1024 ** 2)
        logger.info(f"Screening set size: {size_mb:.2f} MB")


if __name__ == '__main__':
    main()
