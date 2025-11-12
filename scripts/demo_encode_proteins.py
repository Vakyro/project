"""
Demo script for protein encoder.

Shows how to:
1. Load and initialize the ESM2-based protein encoder
2. Encode single and multiple protein sequences
3. Compare sequences using cosine similarity
4. Handle long sequences with chunking
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import csv
from protein_encoder.esm_model import ProteinEncoderESM2
from protein_encoder.utils import encode_long_sequence, get_sequence_stats


def demo_basic_encoding():
    """Basic protein encoding demo."""
    print("=" * 70)
    print("DEMO 1: Basic Protein Encoding")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Initialize encoder with smallest ESM2 model for demo
    print("\nInitializing protein encoder...")
    encoder = ProteinEncoderESM2(
        plm_name="facebook/esm2_t12_35M_UR50D",  # Smallest model for demo
        pooling="attention",
        proj_dim=256,
        dropout=0.1,
        gradient_checkpointing=False
    ).to(device).eval()

    print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Load protein sequences from CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'proteins.csv')
    sequences = []
    sequence_names = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequences.append(row['sequence'])
            sequence_names.append(row['name'])
            # Load first 3 for this demo
            if len(sequences) >= 3:
                break

    # Show sequence stats
    print(f"\n{len(sequences)} protein sequences:")
    for name, seq in zip(sequence_names, sequences):
        print(f"  {name}: {len(seq)} aa")

    # Tokenize and encode
    print("\nEncoding proteins...")
    batch = encoder.tokenize(sequences, max_len=1024)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        embeddings = encoder(batch)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding norms: {torch.norm(embeddings, dim=1)}")

    # Compute similarities
    print("\n" + "-" * 70)
    print("Pairwise Cosine Similarities:")
    print("-" * 70)

    similarity_matrix = embeddings @ embeddings.t()

    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            sim = similarity_matrix[i, j].item()
            print(f"  {sequence_names[i]}")
            print(f"   vs {sequence_names[j]}")
            print(f"   => Similarity: {sim:.4f}\n")

    print("=" * 70)


def demo_long_sequence():
    """Demo handling of long sequences with chunking."""
    print("\n" + "=" * 70)
    print("DEMO 2: Long Sequence Handling")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize encoder
    print("\nInitializing encoder...")
    encoder = ProteinEncoderESM2(
        plm_name="facebook/esm2_t12_35M_UR50D",
        pooling="attention",
        proj_dim=256
    ).to(device).eval()

    # Simulate a long sequence (2000 aa)
    long_seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK" * 10

    print(f"\nSequence length: {len(long_seq)} aa")
    print("This exceeds ESM2's context window (~1024 tokens)")

    # Encode with chunking
    print("\nEncoding with chunking...")
    embedding = encode_long_sequence(
        encoder,
        long_seq,
        device=device,
        max_len=1000,
        overlap=50
    )

    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {torch.norm(embedding).item():.4f}")

    print("\nLong sequences are automatically chunked and averaged!")
    print("=" * 70)


def demo_pooling_comparison():
    """Compare different pooling strategies."""
    print("\n" + "=" * 70)
    print("DEMO 3: Pooling Strategy Comparison")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test sequence
    test_seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

    pooling_methods = ["attention", "mean", "cls"]
    embeddings = {}

    for pooling in pooling_methods:
        print(f"\nTesting {pooling} pooling...")

        encoder = ProteinEncoderESM2(
            plm_name="facebook/esm2_t12_35M_UR50D",
            pooling=pooling,
            proj_dim=256
        ).to(device).eval()

        batch = encoder.tokenize([test_seq])
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            z = encoder(batch)

        embeddings[pooling] = z
        print(f"  Embedding norm: {torch.norm(z).item():.4f}")

    # Compare embeddings
    print("\n" + "-" * 70)
    print("Cross-pooling Similarities:")
    print("-" * 70)

    for i, pool1 in enumerate(pooling_methods):
        for pool2 in pooling_methods[i + 1:]:
            sim = (embeddings[pool1] @ embeddings[pool2].t()).item()
            print(f"  {pool1} vs {pool2}: {sim:.4f}")

    print("\nAll pooling methods produce similar (but not identical) embeddings")
    print("Attention pooling can learn to focus on important regions")
    print("=" * 70)


def demo_batch_processing():
    """Demo efficient batch processing."""
    print("\n" + "=" * 70)
    print("DEMO 4: Batch Processing")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = ProteinEncoderESM2(
        plm_name="facebook/esm2_t12_35M_UR50D",
        pooling="attention",
        proj_dim=256
    ).to(device).eval()

    # Load multiple short sequences from CSV (peptides)
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'proteins.csv')
    sequences = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Load only short sequences (peptides)
            if int(row['length']) <= 50:
                sequences.append(row['sequence'])
                if len(sequences) >= 5:
                    break

    print(f"\nEncoding {len(sequences)} sequences from CSV...")

    # Use the convenience encode method
    embeddings = encoder.encode(
        sequences,
        device=device,
        batch_size=2  # Process 2 at a time
    )

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Average similarity: {(embeddings @ embeddings.t()).mean().item():.4f}")

    print("\nBatch processing is efficient for multiple sequences!")
    print("=" * 70)


if __name__ == "__main__":
    print("\n")
    print("#" * 70)
    print("# PROTEIN ENCODER DEMO")
    print("#" * 70)

    try:
        # Run all demos
        demo_basic_encoding()
        demo_long_sequence()
        demo_pooling_comparison()
        demo_batch_processing()

        print("\n" + "#" * 70)
        print("# ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("#" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
