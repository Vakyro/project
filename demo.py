"""
CLIPZyme Unified Demo

Demonstrates all features of the refactored CLIPZyme system:
- Simple Facade API
- Configuration system
- Builder pattern
- Repository pattern
- Factory pattern

This replaces the 10+ scattered demo scripts with one comprehensive example.
"""

import argparse
from pathlib import Path

# Import the new Facade API
from clipzyme import CLIPZyme


def demo_simple_api():
    """Demo 1: Simplest possible usage with Facade."""
    print("\n" + "=" * 70)
    print("DEMO 1: Simple API Usage (Facade Pattern)")
    print("=" * 70)

    # One-line initialization
    clipzyme = CLIPZyme()  # Uses default configuration

    print(f"\n{clipzyme}")

    # Load some data from CSV
    proteins = clipzyme.load_proteins_from_csv(max_length=100)
    reactions = clipzyme.load_reactions_from_csv()

    print(f"\nLoaded {len(proteins)} proteins and {len(reactions)} reactions from CSV")

    # Get sequences
    protein_seqs = [p.sequence for p in proteins[:3]]
    reaction_smiles = [r.reaction_smiles for r in reactions[:3]]

    # Compute similarity (one function call!)
    similarity = clipzyme.compute_similarity(protein_seqs, reaction_smiles)

    print(f"\nSimilarity matrix shape: {similarity.shape}")
    print("\nSimilarity Matrix:")
    print(similarity)

    # Find best matches
    best_reactions = clipzyme.find_best_reactions_for_protein(
        protein_seqs[0],
        reaction_smiles,
        top_k=3
    )

    print(f"\nBest reactions for {proteins[0].name}:")
    for match in best_reactions:
        print(f"  Rank {match['rank']}: {match['reaction'][:40]}... (score: {match['score']:.4f})")

    print("\nThis is the power of the Facade pattern!")


def demo_configuration_system():
    """Demo 2: Using the configuration system."""
    print("\n" + "=" * 70)
    print("DEMO 2: Configuration System")
    print("=" * 70)

    print("\nOption 1: Default configuration")
    clipzyme_default = CLIPZyme(config='default')
    print(f"  Embedding dim: {clipzyme_default.get_embedding_dim()}")

    print("\nOption 2: CLIPZyme faithful configuration")
    # This would use the paper's exact architecture (ESM2-650M + EGNN + DMPNN)
    # Commented out because it requires large model download
    # clipzyme_faithful = CLIPZyme(config='faithful')

    print("\nOption 3: Custom YAML configuration")
    config_path = Path('configs/default.yaml')
    if config_path.exists():
        clipzyme_yaml = CLIPZyme(config=str(config_path))
        print(f"  Loaded from: {config_path}")
        print(f"  Embedding dim: {clipzyme_yaml.get_embedding_dim()}")
    else:
        print(f"  (YAML config not found at {config_path})")

    print("\nConfiguration system enables:")
    print("  - No hardcoded values")
    print("  - Easy experimentation")
    print("  - Reproducible research")


def demo_builder_pattern():
    """Demo 3: Building models with Builder pattern."""
    print("\n" + "=" * 70)
    print("DEMO 3: Builder Pattern")
    print("=" * 70)

    from models.builder import CLIPZymeBuilder
    from config.config import ProteinEncoderConfig, ReactionEncoderConfig

    # Fluent API for building models
    model = (CLIPZymeBuilder()
             .with_protein_encoder_config(ProteinEncoderConfig(
                 type='ESM2',
                 plm_name='facebook/esm2_t12_35M_UR50D',
                 proj_dim=256
             ))
             .with_reaction_encoder_config(ReactionEncoderConfig(
                 type='Enhanced',
                 hidden_dim=128,
                 proj_dim=256
             ))
             .with_temperature(0.07)
             .on_device('cpu')
             .build())

    print(f"\nBuilt model: {type(model).__name__}")
    print(f"  Protein encoder: {type(model.protein_encoder).__name__}")
    print(f"  Reaction encoder: {type(model.reaction_encoder).__name__}")
    print(f"  Temperature: {model.temperature.item():.3f}")

    print("\nBuilder pattern provides:")
    print("  - Fluent, readable API")
    print("  - Step-by-step construction")
    print("  - Flexible configuration")


def demo_repository_pattern():
    """Demo 4: Using Repository pattern for data access."""
    print("\n" + "=" * 70)
    print("DEMO 4: Repository Pattern")
    print("=" * 70)

    from data.repositories import (
        ProteinRepository,
        ReactionRepository,
        EnzymeReactionRepository
    )

    # Proteins
    protein_repo = ProteinRepository('data/proteins.csv')

    print(f"\nTotal proteins: {protein_repo.count()}")
    print(f"Short proteins (<=50 aa): {protein_repo.count(max_length=50)}")
    print(f"GFP proteins: {protein_repo.count(name_contains='GFP')}")

    # Load specific protein
    gfp = protein_repo.load_by_id('GFP-full')
    if gfp:
        print(f"\nLoaded: {gfp.name}")
        print(f"  Length: {gfp.length} aa")
        print(f"  Sequence: {gfp.sequence[:50]}...")

    # Reactions
    reaction_repo = ReactionRepository('data/reactions_extended.csv')

    print(f"\nTotal reactions: {reaction_repo.count()}")
    print(f"Reduction reactions: {reaction_repo.count(name_contains='reduction')}")

    # Random sampling
    random_proteins = protein_repo.get_random_sample(n=3, max_length=100)
    print(f"\nRandom sample of 3 proteins:")
    for p in random_proteins:
        print(f"  - {p.name} ({p.length} aa)")

    print("\nRepository pattern provides:")
    print("  - Clean data access")
    print("  - Filtering and querying")
    print("  - Caching")
    print("  - Decoupled from business logic")


def demo_factory_pattern():
    """Demo 5: Creating encoders with Factory pattern."""
    print("\n" + "=" * 70)
    print("DEMO 5: Factory Pattern")
    print("=" * 70)

    from common.factory import create_protein_encoder, create_reaction_encoder
    from config.config import ProteinEncoderConfig, ReactionEncoderConfig

    # Create encoder from configuration
    protein_config = ProteinEncoderConfig(
        type='ESM2',
        plm_name='facebook/esm2_t12_35M_UR50D',
        proj_dim=256
    )

    protein_encoder = create_protein_encoder(protein_config)

    print(f"\nCreated protein encoder: {type(protein_encoder).__name__}")
    print(f"  Embedding dim: {protein_encoder.get_embedding_dim()}")

    # Try different reaction encoder types
    for encoder_type in ['GNN', 'Enhanced']:
        reaction_config = ReactionEncoderConfig(
            type=encoder_type,
            hidden_dim=128,
            proj_dim=256
        )

        reaction_encoder = create_reaction_encoder(reaction_config)
        print(f"\nCreated {encoder_type} reaction encoder: {type(reaction_encoder).__name__}")

    print("\nFactory pattern provides:")
    print("  - Dynamic creation based on config")
    print("  - No hardcoded instantiation")
    print("  - Easy to add new encoder types")


def demo_complete_workflow():
    """Demo 6: Complete workflow from data to predictions."""
    print("\n" + "=" * 70)
    print("DEMO 6: Complete Workflow")
    print("=" * 70)

    print("\nStep 1: Initialize system")
    clipzyme = CLIPZyme(config='default', device='cpu')

    print("\nStep 2: Load enzyme-reaction pairs from CSV")
    pairs = clipzyme.load_enzyme_reactions_from_csv(max_length=500)
    print(f"  Loaded {len(pairs)} pairs")

    if len(pairs) >= 3:
        print("\nStep 3: Encode proteins and reactions")
        proteins = [p.sequence for p in pairs[:3]]
        reactions = [p.reaction for p in pairs[:3]]

        protein_embeddings = clipzyme.encode_proteins(proteins)
        reaction_embeddings = clipzyme.encode_reactions(reactions)

        print(f"  Protein embeddings: {protein_embeddings.shape}")
        print(f"  Reaction embeddings: {reaction_embeddings.shape}")

        print("\nStep 4: Compute cross-modal similarity")
        similarity = clipzyme.compute_similarity(proteins, reactions)

        print("\n  Similarity Matrix:")
        print("  " + "  ".join([f"R{i+1}" for i in range(len(reactions))]))
        for i, pair in enumerate(pairs[:3]):
            name = pair.name[:20]
            sims = "  ".join([f"{similarity[i, j]:.3f}" for j in range(len(reactions))])
            print(f"  {name:20s}  {sims}")

        print("\nStep 5: Find best matches")
        test_protein = proteins[0]
        matches = clipzyme.find_best_reactions_for_protein(test_protein, reactions, top_k=3)

        print(f"\n  Top reactions for {pairs[0].name}:")
        for match in matches:
            print(f"    {match['rank']}. Score: {match['score']:.4f} - {match['reaction'][:40]}...")

    print("\nComplete workflow in <50 lines of code!")


def main():
    """Run all demos."""
    parser = argparse.ArgumentParser(description='CLIPZyme Unified Demo')
    parser.add_argument(
        '--demo',
        type=str,
        choices=['simple', 'config', 'builder', 'repository', 'factory', 'workflow', 'all'],
        default='all',
        help='Which demo to run'
    )

    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("# CLIPZyme Unified Demo - Refactored Architecture")
    print("#" * 70)
    print("\nShowcasing Design Patterns:")
    print("  - Facade Pattern (Simple API)")
    print("  - Strategy Pattern (Encoder interfaces)")
    print("  - Factory Pattern (Dynamic creation)")
    print("  - Builder Pattern (Fluent construction)")
    print("  - Repository Pattern (Data access)")
    print("  - Configuration System (YAML + dataclasses)")

    try:
        if args.demo == 'all':
            demo_simple_api()
            demo_configuration_system()
            demo_builder_pattern()
            demo_repository_pattern()
            demo_factory_pattern()
            demo_complete_workflow()
        elif args.demo == 'simple':
            demo_simple_api()
        elif args.demo == 'config':
            demo_configuration_system()
        elif args.demo == 'builder':
            demo_builder_pattern()
        elif args.demo == 'repository':
            demo_repository_pattern()
        elif args.demo == 'factory':
            demo_factory_pattern()
        elif args.demo == 'workflow':
            demo_complete_workflow()

        print("\n" + "#" * 70)
        print("# All Demos Completed Successfully!")
        print("#" * 70)
        print("\nNext steps:")
        print("  - Edit configs/default.yaml to customize settings")
        print("  - Add your own data to data/ CSVs")
        print("  - Use CLIPZyme() facade for quick prototyping")
        print("  - Use Builder for advanced model construction")
        print("\n")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
