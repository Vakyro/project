"""
Demo script to test enhanced features and models.

Compares:
1. Original model with basic features
2. Enhanced model with attention pooling and rich features
3. Dual-branch model with change-only graph

Shows how the improvements reduce similarity between different reactions.
"""

import sys
import os
import csv

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from reaction_encoder.chem import parse_reaction_smiles
from reaction_encoder.builder import build_transition_graph
from reaction_encoder.builder_change_only import build_change_only_graph
from reaction_encoder.model import ReactionGNN
from reaction_encoder.model_enhanced import ReactionGNNEnhanced, DualBranchReactionGNN


def load_reactions(csv_path):
    """Load reactions from CSV."""
    reactions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            reactions.append({
                'smiles': row['reaction_smiles'],
                'name': row['reaction_name']
            })
    return reactions


def build_graphs(reactions, use_enhanced_features=False):
    """Build graphs for all reactions."""
    graphs_full = []
    graphs_change = []
    valid_reactions = []

    print(f"\nBuilding graphs (enhanced_features={use_enhanced_features})...")

    for rxn in reactions:
        try:
            reacts, prods = parse_reaction_smiles(rxn['smiles'])

            # Full graph
            data_full = build_transition_graph(reacts, prods, use_enhanced_features)

            # Change-only graph
            data_change = build_change_only_graph(reacts, prods, use_enhanced_features)

            graphs_full.append(data_full)
            graphs_change.append(data_change)
            valid_reactions.append(rxn)

            print(f"  [{len(valid_reactions)}] {rxn['name']}: "
                  f"full={data_full.num_nodes} nodes, "
                  f"change={data_change.num_nodes} nodes")

        except Exception as e:
            print(f"  ERROR: {rxn['name']} - {e}")

    return graphs_full, graphs_change, valid_reactions


def encode_reactions(model, graphs_full, graphs_change=None, model_type='single'):
    """Encode reactions with given model."""
    model.eval()
    embeddings = []

    with torch.no_grad():
        if model_type == 'dual_branch':
            # Dual-branch model
            for g_full, g_change in zip(graphs_full, graphs_change):
                z = model(g_full, g_change)
                embeddings.append(z)
        else:
            # Single-branch model
            for g in graphs_full:
                z = model(g)
                embeddings.append(z)

    return torch.cat(embeddings, dim=0)


def compute_similarities(embeddings, reactions):
    """Compute and display pairwise similarities."""
    similarity_matrix = embeddings @ embeddings.t()

    print("\nPairwise Cosine Similarities:")
    print("-" * 60)

    similarities = []
    for i in range(len(reactions)):
        for j in range(i + 1, len(reactions)):
            sim = similarity_matrix[i, j].item()
            similarities.append(sim)
            print(f"  [{i+1}] {reactions[i]['name']}")
            print(f"   vs [{j+1}] {reactions[j]['name']}")
            print(f"   => Similarity: {sim:.4f}\n")

    # Statistics
    avg_sim = sum(similarities) / len(similarities) if similarities else 0
    max_sim = max(similarities) if similarities else 0
    min_sim = min(similarities) if similarities else 0

    print("-" * 60)
    print(f"Average: {avg_sim:.4f}, Max: {max_sim:.4f}, Min: {min_sim:.4f}")
    print("-" * 60)

    return avg_sim


def demo_comparison():
    """Compare original vs enhanced models."""
    print("=" * 70)
    print("DEMO: Comparing Original vs Enhanced Models")
    print("=" * 70)

    # Load reactions
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reactions.csv')
    reactions = load_reactions(csv_path)
    print(f"\nLoaded {len(reactions)} reactions")

    # ========== MODEL 1: Original ==========
    print("\n" + "=" * 70)
    print("MODEL 1: Original ReactionGNN (basic features, mean pooling)")
    print("=" * 70)

    graphs_full, graphs_change, valid_rxns = build_graphs(reactions, use_enhanced_features=False)

    if not graphs_full:
        print("No valid graphs!")
        return

    x_dim = graphs_full[0].x.size(1)
    e_dim = graphs_full[0].edge_attr.size(1)

    model1 = ReactionGNN(x_dim=x_dim, e_dim=e_dim, hidden=128, layers=3, out_dim=256)
    print(f"\nModel parameters: {sum(p.numel() for p in model1.parameters()):,}")

    embeddings1 = encode_reactions(model1, graphs_full)
    avg_sim1 = compute_similarities(embeddings1, valid_rxns)

    # ========== MODEL 2: Enhanced Features + Attention ==========
    print("\n" + "=" * 70)
    print("MODEL 2: Enhanced Features + Attention Pooling")
    print("=" * 70)

    graphs_full_enh, graphs_change_enh, _ = build_graphs(reactions, use_enhanced_features=True)

    x_dim_enh = graphs_full_enh[0].x.size(1)

    model2 = ReactionGNNEnhanced(
        x_dim=x_dim_enh,
        e_dim=e_dim,
        hidden=128,
        layers=3,
        out_dim=256,
        dropout=0.2,
        use_attention=True
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model2.parameters()):,}")

    embeddings2 = encode_reactions(model2, graphs_full_enh)
    avg_sim2 = compute_similarities(embeddings2, valid_rxns)

    # ========== MODEL 3: Dual-Branch ==========
    print("\n" + "=" * 70)
    print("MODEL 3: Dual-Branch (Full + Change-Only)")
    print("=" * 70)

    model3 = DualBranchReactionGNN(
        x_dim=x_dim_enh,
        e_dim=e_dim,
        hidden=128,
        layers=3,
        out_dim=256,
        dropout=0.2
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model3.parameters()):,}")

    embeddings3 = encode_reactions(
        model3,
        graphs_full_enh,
        graphs_change_enh,
        model_type='dual_branch'
    )
    avg_sim3 = compute_similarities(embeddings3, valid_rxns)

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("SUMMARY: Average Cosine Similarities")
    print("=" * 70)
    print(f"Model 1 (Original):              {avg_sim1:.4f}")
    print(f"Model 2 (Enhanced + Attention):  {avg_sim2:.4f}")
    print(f"Model 3 (Dual-Branch):           {avg_sim3:.4f}")
    print()
    print("Lower average similarity = better differentiation between reactions")
    print("=" * 70)


if __name__ == "__main__":
    try:
        demo_comparison()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
