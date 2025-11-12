"""
Demo script showing how to encode reactions from a CSV file.
"""

import sys
import os
import csv

# Add parent directory to path to import reaction_encoder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from reaction_encoder.chem import parse_reaction_smiles
from reaction_encoder.builder import build_transition_graph
from reaction_encoder.model import ReactionGNN


def load_reactions_from_csv(csv_path):
    """Load reactions from a CSV file."""
    reactions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            reactions.append({
                'smiles': row['reaction_smiles'],
                'name': row['reaction_name']
            })
    return reactions


def demo_csv_reactions():
    """Encode reactions from CSV and display embeddings."""
    print("=" * 60)
    print("Demo: Encoding Reactions from CSV")
    print("=" * 60)

    # Path to CSV file
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reactions.csv')

    if not os.path.exists(csv_path):
        print(f"\nError: CSV file not found at {csv_path}")
        return

    # Load reactions
    print(f"\nLoading reactions from: {csv_path}")
    reactions = load_reactions_from_csv(csv_path)
    print(f"Loaded {len(reactions)} reactions")

    # Build graphs
    print("\n" + "-" * 60)
    print("Building transition state graphs...")
    print("-" * 60)

    graphs = []
    valid_reactions = []

    for i, rxn in enumerate(reactions):
        try:
            print(f"\n[{i+1}] {rxn['name']}")
            print(f"    SMILES: {rxn['smiles']}")

            reacts, prods = parse_reaction_smiles(rxn['smiles'])
            data = build_transition_graph(reacts, prods)

            print(f"    Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}")

            graphs.append(data)
            valid_reactions.append(rxn)

        except Exception as e:
            print(f"    ERROR: {e}")

    if not graphs:
        print("\nNo valid reactions to encode!")
        return

    print(f"\n{len(graphs)} reactions successfully parsed")

    # Initialize model
    print("\n" + "-" * 60)
    print("Initializing ReactionGNN...")
    print("-" * 60)

    x_dim = graphs[0].x.size(1)
    e_dim = graphs[0].edge_attr.size(1)

    model = ReactionGNN(
        x_dim=x_dim,
        e_dim=e_dim,
        hidden=128,
        layers=3,
        out_dim=256
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Encode all reactions
    print("\n" + "-" * 60)
    print("Encoding reactions...")
    print("-" * 60)

    model.eval()
    embeddings = []

    with torch.no_grad():
        for i, data in enumerate(graphs):
            z = model(data)
            embeddings.append(z)
            print(f"[{i+1}] {valid_reactions[i]['name']}")
            print(f"    Embedding shape: {z.shape}, Norm: {torch.norm(z).item():.4f}")

    embeddings = torch.cat(embeddings, dim=0)

    # Compute pairwise similarities
    print("\n" + "-" * 60)
    print("Pairwise Cosine Similarities:")
    print("-" * 60)

    similarity_matrix = embeddings @ embeddings.t()

    for i in range(len(valid_reactions)):
        for j in range(i + 1, len(valid_reactions)):
            sim = similarity_matrix[i, j].item()
            print(f"[{i+1}] {valid_reactions[i]['name']}")
            print(f" vs [{j+1}] {valid_reactions[j]['name']}")
            print(f" => Similarity: {sim:.4f}\n")

    print("=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        demo_csv_reactions()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
