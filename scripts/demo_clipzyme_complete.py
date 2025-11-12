"""
Complete CLIPZyme Demo: Protein + Reaction Encoders Together

Demonstrates the full pipeline:
1. Encode proteins with ESM2
2. Encode reactions with ReactionGNN
3. Compute cross-modal similarities
4. Show how to use CLIP loss for training (conceptual)

This is the foundation for enzyme-reaction matching!
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import csv

# Protein encoder
from protein_encoder.esm_model import ProteinEncoderESM2

# Reaction encoder
from reaction_encoder.chem import parse_reaction_smiles
from reaction_encoder.builder import build_transition_graph
from reaction_encoder.model_enhanced import ReactionGNNEnhanced

# Loss function
from reaction_encoder.loss import clip_loss


def demo_complete_pipeline():
    """Complete CLIPZyme pipeline demonstration."""
    print("=" * 70)
    print("COMPLETE CLIPZYME DEMO: Protein <-> Reaction Matching")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # ========== Setup Encoders ==========
    print("\n" + "-" * 70)
    print("1. Initializing Encoders")
    print("-" * 70)

    # Protein Encoder
    print("\n[Protein Encoder]")
    protein_encoder = ProteinEncoderESM2(
        plm_name="facebook/esm2_t12_35M_UR50D",  # Smallest ESM2 for demo
        pooling="attention",
        proj_dim=256,
        dropout=0.1
    ).to(device).eval()
    print(f"  Parameters: {sum(p.numel() for p in protein_encoder.parameters()):,}")

    # Reaction Encoder
    print("\n[Reaction Encoder]")
    # Load a sample reaction to get dimensions
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reactions.csv')
    sample_reactions = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_reactions.append(row['reaction_smiles'])
            if len(sample_reactions) >= 1:
                break

    reacts, prods = parse_reaction_smiles(sample_reactions[0])
    sample_data = build_transition_graph(reacts, prods, use_enhanced_features=True)

    reaction_encoder = ReactionGNNEnhanced(
        x_dim=sample_data.x.size(1),
        e_dim=sample_data.edge_attr.size(1),
        hidden=128,
        layers=3,
        out_dim=256,
        dropout=0.1,
        use_attention=True
    ).to(device).eval()
    print(f"  Parameters: {sum(p.numel() for p in reaction_encoder.parameters()):,}")

    # ========== Prepare Data ==========
    print("\n" + "-" * 70)
    print("2. Preparing Enzyme-Reaction Pairs")
    print("-" * 70)

    # Load enzyme-reaction pairs from CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'enzyme_reactions.csv')
    enzyme_reaction_pairs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            enzyme_reaction_pairs.append({
                'enzyme': row['name'],
                'sequence': row['sequence'],
                'reaction': row['reaction'],
                'reaction_name': row['description']
            })
            # Load first 3 for this demo
            if len(enzyme_reaction_pairs) >= 3:
                break

    print(f"\n{len(enzyme_reaction_pairs)} enzyme-reaction pairs loaded from CSV:")
    for pair in enzyme_reaction_pairs:
        print(f"  - {pair['enzyme']} -> {pair['reaction_name']}")

    # ========== Encode Proteins ==========
    print("\n" + "-" * 70)
    print("3. Encoding Proteins")
    print("-" * 70)

    sequences = [pair["sequence"] for pair in enzyme_reaction_pairs]
    batch_prot = protein_encoder.tokenize(sequences, max_len=1024)
    batch_prot = {k: v.to(device) for k, v in batch_prot.items()}

    with torch.no_grad():
        protein_embeddings = protein_encoder(batch_prot)

    print(f"Protein embeddings: {protein_embeddings.shape}")

    # ========== Encode Reactions ==========
    print("\n" + "-" * 70)
    print("4. Encoding Reactions")
    print("-" * 70)

    reaction_embeddings_list = []
    for pair in enzyme_reaction_pairs:
        reacts, prods = parse_reaction_smiles(pair["reaction"])
        data = build_transition_graph(reacts, prods, use_enhanced_features=True)
        data = data.to(device)

        with torch.no_grad():
            z = reaction_encoder(data)
            reaction_embeddings_list.append(z)

    reaction_embeddings = torch.cat(reaction_embeddings_list, dim=0)
    print(f"Reaction embeddings: {reaction_embeddings.shape}")

    # ========== Cross-Modal Similarity ==========
    print("\n" + "-" * 70)
    print("5. Computing Cross-Modal Similarities")
    print("-" * 70)

    # Compute similarity matrix: (proteins x reactions)
    similarity_matrix = protein_embeddings @ reaction_embeddings.t()

    print("\nProtein-Reaction Similarity Matrix:")
    print("(Rows=Proteins, Cols=Reactions)\n")

    # Header
    print(" " * 25 + "  ".join([f"R{i+1}" for i in range(len(enzyme_reaction_pairs))]))

    for i, pair in enumerate(enzyme_reaction_pairs):
        enzyme_name = pair["enzyme"][:23]
        sims = [f"{similarity_matrix[i, j].item():.3f}" for j in range(len(enzyme_reaction_pairs))]
        print(f"{enzyme_name:23s}  " + "  ".join(sims))

    print("\nDiagonal values (correct pairs) should be highest after training!")

    # ========== Show Retrieval ==========
    print("\n" + "-" * 70)
    print("6. Enzyme Retrieval for Reaction")
    print("-" * 70)

    # For each reaction, find most similar enzyme
    print("\nFor each reaction, top matching enzyme:")
    for j, pair in enumerate(enzyme_reaction_pairs):
        scores = similarity_matrix[:, j]
        top_idx = torch.argmax(scores).item()
        top_score = scores[top_idx].item()

        print(f"\n  Reaction: {pair['reaction_name']}")
        print(f"  Top enzyme: {enzyme_reaction_pairs[top_idx]['enzyme']}")
        print(f"  Score: {top_score:.4f}")
        correct = "CORRECT!" if top_idx == j else "Wrong (untrained model)"
        print(f"  {correct}")

    # ========== CLIP Loss Demonstration ==========
    print("\n" + "-" * 70)
    print("7. Contrastive Loss (CLIP) Demonstration")
    print("-" * 70)

    # Compute CLIP loss (just for demonstration, not training)
    loss = clip_loss(protein_embeddings, reaction_embeddings, temperature=0.07)
    print(f"\nCLIP Loss: {loss.item():.4f}")
    print("\nDuring training, this loss would:")
    print("  - Push matching protein-reaction pairs together")
    print("  - Pull non-matching pairs apart")
    print("  - Eventually enable accurate enzyme-reaction retrieval")

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n[OK] Protein encoder: ESM2 -> Attention Pool -> Projection -> L2 norm")
    print("[OK] Reaction encoder: GNN -> Attention Pool -> Projection -> L2 norm")
    print("[OK] Both produce 256-dim embeddings in shared space")
    print("[OK] Cosine similarity measures protein-reaction compatibility")
    print("[OK] CLIP loss trains the model to align correct pairs")
    print("\nWith proper training on enzyme-reaction datasets,")
    print("this system can:")
    print("  - Match enzymes to reactions they catalyze")
    print("  - Virtual screen enzyme databases for a given reaction")
    print("  - Discover novel enzymes for desired transformations")
    print("\nThis is the core of CLIPZyme!")
    print("=" * 70)


if __name__ == "__main__":
    print("\n")
    print("#" * 70)
    print("# CLIPZYME: Complete Enzyme-Reaction Encoder Demo")
    print("#" * 70)

    try:
        demo_complete_pipeline()

        print("\n" + "#" * 70)
        print("# DEMO COMPLETED SUCCESSFULLY!")
        print("#" * 70)
        print()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
