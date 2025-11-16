#!/usr/bin/env python3
"""
CLIPZyme Screening Script

Run virtual screening of enzymes against chemical reactions.

Usage:
    # Interactive mode
    python scripts/run_screening.py --config configs/screening_interactive.yaml

    # Batched mode
    python scripts/run_screening.py --config configs/screening_batched.yaml

    # Quick test with single reaction
    python scripts/run_screening.py --reaction "[C:1]=[O:2]>>[C:1]-[O:2]" --model models/clipzyme.pt
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

from screening.screening_set import ScreeningSet, ProteinDatabase, build_screening_set_from_model
from screening.interactive_mode import InteractiveScreener, InteractiveScreeningConfig
from screening.batched_mode import BatchedScreener, BatchedScreeningConfig
from screening.cache import EmbeddingCache
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
    """
    Load CLIPZyme model from checkpoint or build from config.

    Args:
        config: Model configuration dictionary

    Returns:
        Loaded CLIPZyme model
    """
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Load from checkpoint if provided
    if config.get('checkpoint_path'):
        logger.info(f"Loading model from checkpoint: {config['checkpoint_path']}")
        model = torch.load(config['checkpoint_path'], map_location=device)
        model = model.to(device)
        model.eval()
        return model

    # Build from config
    if config.get('config_path'):
        logger.info(f"Building model from config: {config['config_path']}")
        clipzyme_config = CLIPZymeConfig.from_yaml(config['config_path'])

        builder = CLIPZymeBuilder()
        model = (builder
                 .with_config(clipzyme_config)
                 .build())

        model = model.to(device)
        model.eval()
        return model

    raise ValueError("Must provide either checkpoint_path or config_path")


def load_screening_set(config: dict, model=None) -> ScreeningSet:
    """
    Load or build screening set.

    Args:
        config: Screening set configuration
        model: Optional model for building screening set from scratch

    Returns:
        ScreeningSet with pre-embedded proteins
    """
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Load from pickle file if provided
    if config.get('embeddings_path'):
        logger.info(f"Loading screening set from: {config['embeddings_path']}")
        screening_set = ScreeningSet(device=device)
        screening_set.load_from_pickle(config['embeddings_path'], device=device)
        return screening_set

    # Build from CSV if requested
    if config.get('build_from_csv') and config.get('csv_path'):
        if model is None:
            raise ValueError("Model required to build screening set from CSV")

        logger.info(f"Building screening set from CSV: {config['csv_path']}")

        # Load protein database
        protein_db = ProteinDatabase()
        protein_db.load_from_csv(
            csv_path=config['csv_path'],
            id_column=config.get('id_column', 'protein_id'),
            sequence_column=config.get('sequence_column', 'sequence')
        )

        # Build screening set
        screening_set = build_screening_set_from_model(
            model=model,
            protein_database=protein_db,
            batch_size=config.get('batch_size', 32),
            device=device,
            show_progress=True
        )

        # Save if output path provided
        if config.get('output_path'):
            screening_set.save_to_pickle(config['output_path'])

        return screening_set

    raise ValueError("Must provide embeddings_path or build_from_csv=true with csv_path")


def load_protein_database(config: dict) -> ProteinDatabase:
    """Load protein database for metadata."""
    if config.get('csv_path'):
        logger.info(f"Loading protein database from: {config['csv_path']}")
        protein_db = ProteinDatabase()
        protein_db.load_from_csv(
            csv_path=config['csv_path'],
            id_column=config.get('id_column', 'protein_id'),
            sequence_column=config.get('sequence_column', 'sequence'),
            metadata_columns=config.get('metadata_columns', [])
        )
        return protein_db

    if config.get('pickle_path'):
        logger.info(f"Loading protein database from: {config['pickle_path']}")
        protein_db = ProteinDatabase()
        protein_db.load_from_pickle(config['pickle_path'])
        return protein_db

    return None


def setup_cache(config: dict) -> EmbeddingCache:
    """Setup embedding cache."""
    if not config.get('enabled', False):
        return None

    cache = EmbeddingCache(
        memory_cache_size=config.get('memory_cache_size', 1000),
        disk_cache_dir=config.get('disk_cache_dir'),
        disk_cache_size_gb=config.get('disk_cache_size_gb', 10.0),
        use_memory=config.get('memory_cache_size', 0) > 0,
        use_disk=config.get('disk_cache_dir') is not None
    )

    logger.info(f"Cache enabled: {cache}")
    return cache


def run_interactive_screening(config: dict):
    """Run interactive screening mode."""
    logger.info("=== Interactive Screening Mode ===")

    # Load components
    model = load_model(config['model'])
    screening_set = load_screening_set(config['screening_set'], model)
    protein_db = load_protein_database(config.get('protein_database', {}))
    cache = setup_cache(config.get('cache', {}))

    # Create screener
    screener_config = InteractiveScreeningConfig(
        top_k=config['screening'].get('top_k', 100),
        device=config['model'].get('device', 'cuda'),
        evaluate=config['screening'].get('evaluate', False)
    )

    screener = InteractiveScreener(
        model=model,
        screening_set=screening_set,
        protein_database=protein_db,
        config=screener_config
    )

    logger.info(f"Screener initialized: {screener}")

    # Get reactions to screen
    reactions = config.get('reactions', {})

    if reactions.get('smiles_list'):
        # Screen from list
        reaction_smiles_list = reactions['smiles_list']
        reaction_ids = reactions.get('ids_list')
        true_positives_list = reactions.get('true_positives_list')

        logger.info(f"Screening {len(reaction_smiles_list)} reactions...")

        results = screener.screen_reactions(
            reaction_smiles_list=reaction_smiles_list,
            reaction_ids=reaction_ids,
            true_positives_list=true_positives_list,
            top_k=screener_config.top_k,
            show_progress=True
        )

    elif reactions.get('csv_path'):
        # Screen from CSV
        logger.info(f"Screening from CSV: {reactions['csv_path']}")

        import pandas as pd
        df = pd.read_csv(reactions['csv_path'])

        reaction_smiles_list = df[reactions.get('reaction_column', 'reaction_smiles')].tolist()
        reaction_ids = df[reactions.get('id_column', 'reaction_id')].tolist() if reactions.get('id_column') else None

        results = screener.screen_reactions(
            reaction_smiles_list=reaction_smiles_list,
            reaction_ids=reaction_ids,
            top_k=screener_config.top_k,
            show_progress=True
        )

    else:
        logger.error("No reactions provided. Specify smiles_list or csv_path in config.")
        return

    # Save results
    output_dir = Path(config.get('output', {}).get('output_dir', 'results/screening'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save rankings
    import json
    for result in results:
        output_file = output_dir / f"{result.reaction_id}_ranking.json"
        with open(output_file, 'w') as f:
            json.dump({
                'reaction_id': result.reaction_id,
                'top_proteins': result.ranked_protein_ids[:screener_config.top_k],
                'scores': result.scores.cpu().tolist(),
                'metrics': result.metrics
            }, f, indent=2)

    logger.info(f"Results saved to {output_dir}")

    # Print summary
    if results and results[0].metrics:
        from screening.ranking import batch_evaluate_screening
        aggregate_metrics = batch_evaluate_screening(results)
        logger.info(f"\n=== Aggregate Metrics ===")
        for key, value in aggregate_metrics.items():
            logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


def run_batched_screening(config: dict):
    """Run batched screening mode."""
    logger.info("=== Batched Screening Mode ===")

    # Load components
    model = load_model(config['model'])
    screening_set = load_screening_set(config['screening_set'], model)
    protein_db = load_protein_database(config.get('protein_database', {}))

    # Create screener
    screener_config = BatchedScreeningConfig(
        batch_size=config['screening'].get('batch_size', 64),
        num_workers=config['screening'].get('num_workers', 4),
        top_k=config['screening'].get('top_k', 100),
        devices=config['model'].get('devices', ['cuda:0']),
        save_embeddings=config['output'].get('save_embeddings', False),
        save_scores=config['output'].get('save_scores', True),
        output_dir=config['output'].get('output_dir'),
        evaluate=config['screening'].get('evaluate', False)
    )

    screener = BatchedScreener(
        model=model,
        screening_set=screening_set,
        protein_database=protein_db,
        config=screener_config
    )

    logger.info(f"Screener initialized: {screener}")

    # Screen from CSV
    reactions_config = config.get('reactions', {})
    if reactions_config.get('csv_path'):
        logger.info(f"Screening from CSV: {reactions_config['csv_path']}")

        results = screener.screen_from_csv(
            csv_path=reactions_config['csv_path'],
            reaction_column=reactions_config.get('reaction_column', 'reaction_smiles'),
            id_column=reactions_config.get('id_column'),
            true_positives_column=reactions_config.get('true_positives_column'),
            show_progress=True
        )

    elif reactions_config.get('smiles_list'):
        # Screen from list
        results = screener.screen_reactions(
            reaction_smiles_list=reactions_config['smiles_list'],
            reaction_ids=reactions_config.get('ids_list'),
            true_positives_list=reactions_config.get('true_positives_list'),
            show_progress=True
        )

    else:
        logger.error("No reactions provided. Specify csv_path or smiles_list in config.")
        return

    logger.info(f"Screening complete. Processed {len(results)} reactions.")

    # Print summary
    if results and results[0].metrics:
        from screening.ranking import batch_evaluate_screening
        aggregate_metrics = batch_evaluate_screening(results)
        logger.info(f"\n=== Aggregate Metrics ===")
        for key, value in aggregate_metrics.items():
            logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="CLIPZyme Virtual Screening")

    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--reaction',
        type=str,
        help='Single reaction SMILES for quick test'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Path to model checkpoint (for quick test)'
    )

    parser.add_argument(
        '--screening-set',
        type=str,
        help='Path to screening set pickle file (for quick test)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top matches to return (default: 10)'
    )

    args = parser.parse_args()

    # Quick test mode
    if args.reaction:
        if not args.model or not args.screening_set:
            logger.error("--model and --screening-set required for quick test")
            return

        logger.info("=== Quick Test Mode ===")
        logger.info(f"Reaction: {args.reaction}")

        # Load model and screening set
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.load(args.model, map_location=device)
        model.eval()

        screening_set = ScreeningSet(device=device)
        screening_set.load_from_pickle(args.screening_set)

        # Screen
        screener = InteractiveScreener(model=model, screening_set=screening_set)
        result = screener.screen_reaction(args.reaction, top_k=args.top_k)

        # Print results
        logger.info(f"\nTop {args.top_k} matches:")
        for i, (protein_id, score) in enumerate(zip(result.ranked_protein_ids, result.scores), 1):
            logger.info(f"{i}. {protein_id}: {score:.4f}")

        return

    # Config-based mode
    if not args.config:
        parser.print_help()
        logger.error("\nError: --config required (or use --reaction for quick test)")
        return

    # Load config
    config = load_config(args.config)

    # Run appropriate mode
    mode = config['screening'].get('mode', 'interactive')

    if mode == 'interactive':
        run_interactive_screening(config)
    elif mode == 'batched':
        run_batched_screening(config)
    else:
        logger.error(f"Unknown mode: {mode}. Must be 'interactive' or 'batched'")


if __name__ == '__main__':
    main()
