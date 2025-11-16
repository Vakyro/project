"""
Demo script for CLIPZyme inference API.

Shows how to use the simple predictor interface.
"""

import sys
sys.path.insert(0, str(__file__).rsplit('scripts', 1)[0].rstrip('/\\'))

from inference import CLIPZymePredictor, batch_screen_reactions


def demo_simple_prediction():
    """Demo simple prediction with predictor."""
    print("=" * 60)
    print("CLIPZyme Inference Demo - Simple Prediction")
    print("=" * 60)

    # Create predictor from checkpoint
    print("\n[1/3] Loading model...")
    predictor = CLIPZymePredictor.from_checkpoint(
        'checkpoints/clipzyme.pt',  # Replace with actual checkpoint
        device='cpu'  # Use 'cuda' if GPU available
    )
    print("   ✓ Model loaded")

    # Load screening set
    print("\n[2/3] Loading screening set...")
    predictor.load_screening_set('data/screening_set.pkl')
    print("   ✓ Screening set loaded")

    # Screen a reaction
    print("\n[3/3] Screening reaction...")
    reaction = "[C:1]=[O:2]>>[C:1][O:2]"

    result = predictor.screen(reaction, top_k=10)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Reaction: {result.reaction_smiles}")
    print(f"\nTop {len(result.top_proteins)} proteins:")

    for i, (protein, score) in enumerate(zip(result.top_proteins, result.scores), 1):
        print(f"  {i}. {protein}: {score:.4f}")

    print("\n" + "=" * 60)


def demo_batch_screening():
    """Demo batch screening."""
    print("=" * 60)
    print("CLIPZyme Inference Demo - Batch Screening")
    print("=" * 60)

    reactions = [
        "[C:1]=[O:2]>>[C:1][O:2]",
        "[N:1]=[N:2]>>[N:1][N:2]",
        "[C:1]=[C:2]>>[C:1][C:2]"
    ]

    print(f"\nScreening {len(reactions)} reactions...")

    results = batch_screen_reactions(
        checkpoint_path='checkpoints/clipzyme.pt',
        reactions=reactions,
        screening_set_path='data/screening_set.pkl',
        top_k=5,
        batch_size=32,
        device='cpu'
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        print(f"\nReaction {i}: {result.reaction_smiles}")
        print(f"  Top protein: {result.top_proteins[0]} (score: {result.scores[0]:.4f})")

    print("\n" + "=" * 60)


def demo_embeddings():
    """Demo encoding proteins and reactions."""
    print("=" * 60)
    print("CLIPZyme Inference Demo - Embeddings")
    print("=" * 60)

    # Create predictor
    print("\n[1/2] Loading model...")
    predictor = CLIPZymePredictor.from_checkpoint(
        'checkpoints/clipzyme.pt',
        device='cpu'
    )
    print("   ✓ Model loaded")

    # Encode protein
    print("\n[2/2] Encoding protein and reaction...")
    protein_seq = "MSKQLIVNLLKQNNYKNSGSS"
    reaction = "[C:1]=[O:2]>>[C:1][O:2]"

    protein_emb = predictor.encode_protein(protein_seq)
    reaction_emb = predictor.encode_reaction(reaction)

    print(f"\nProtein embedding shape: {protein_emb.shape}")
    print(f"Reaction embedding shape: {reaction_emb.shape}")

    # Compute similarity
    similarity = predictor.compute_similarity(protein_seq, reaction)

    print(f"\nSimilarity: {similarity:.4f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CLIPZyme inference demo')
    parser.add_argument(
        '--demo',
        choices=['simple', 'batch', 'embeddings', 'all'],
        default='simple',
        help='Which demo to run'
    )

    args = parser.parse_args()

    try:
        if args.demo == 'simple' or args.demo == 'all':
            demo_simple_prediction()
            print()

        if args.demo == 'batch' or args.demo == 'all':
            demo_batch_screening()
            print()

        if args.demo == 'embeddings' or args.demo == 'all':
            demo_embeddings()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nNote: This demo requires:")
        print("  - A trained checkpoint at 'checkpoints/clipzyme.pt'")
        print("  - A screening set at 'data/screening_set.pkl'")
        print("\nRun the screening system demos first to generate these files.")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
