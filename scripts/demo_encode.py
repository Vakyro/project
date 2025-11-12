"""
Demo script showing how to encode reactions with the ReactionGNN.
"""

import sys
import os

# Add parent directory to path to import reaction_encoder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import csv
from reaction_encoder.chem import parse_reaction_smiles
from reaction_encoder.builder import build_transition_graph
from reaction_encoder.model import ReactionGNN


def demo_single_reaction():
    """Encode a single reaction and display the embedding."""
    print("=" * 60)
    print("Demo: Encoding a Single Reaction")
    print("=" * 60)

    # Load a reaction from CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reactions_extended.csv')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'Azo reduction' in row['reaction_name']:
                rxn = row['reaction_smiles']
                break

    print(f"\nReaction SMILES: {rxn}")

    # Parse reaction
    print("\n1. Parsing reaction...")
    reacts, prods = parse_reaction_smiles(rxn)
    print(f"   - Found {len(reacts)} reactant(s)")
    print(f"   - Found {len(prods)} product(s)")

    # Build transition graph
    print("\n2. Building transition state graph...")
    data = build_transition_graph(reacts, prods)
    print(f"   - Nodes: {data.num_nodes}")
    print(f"   - Edges: {data.edge_index.size(1)}")
    print(f"   - Node feature dim: {data.x.size(1)}")
    print(f"   - Edge feature dim: {data.edge_attr.size(1)}")

    # Initialize model
    print("\n3. Initializing ReactionGNN...")
    model = ReactionGNN(
        x_dim=data.x.size(1),
        e_dim=data.edge_attr.size(1),
        hidden=128,
        layers=3,
        out_dim=256
    )
    print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Encode reaction
    print("\n4. Encoding reaction...")
    model.eval()
    with torch.no_grad():
        z = model(data)

    print(f"   - Embedding shape: {z.shape}")
    print(f"   - Embedding norm: {torch.norm(z).item():.4f} (should be ~1.0)")
    print(f"   - First 10 values: {z[0, :10].tolist()}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


def demo_multiple_reactions():
    """Encode multiple reactions and compute similarity."""
    print("\n" + "=" * 60)
    print("Demo: Comparing Multiple Reactions")
    print("=" * 60)

    # Load reactions from CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reactions_extended.csv')
    reactions = []
    reaction_names = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Load specific reactions for comparison
            if any(x in row['reaction_name'] for x in ['Carbonyl', 'N=N', 'Triple']):
                reactions.append(row['reaction_smiles'])
                reaction_names.append(row['reaction_name'])
                if len(reactions) >= 3:
                    break

    print(f"\nEncoding {len(reactions)} reactions from CSV...")

    # Build graphs
    graphs = []
    for i, (rxn, name) in enumerate(zip(reactions, reaction_names)):
        try:
            reacts, prods = parse_reaction_smiles(rxn)
            data = build_transition_graph(reacts, prods)
            graphs.append(data)
            print(f"  [{i+1}] {name}: {rxn}")
        except Exception as e:
            print(f"  [{i+1}] {name}: Failed - {e}")

    # Initialize model
    x_dim = graphs[0].x.size(1)
    e_dim = graphs[0].edge_attr.size(1)
    model = ReactionGNN(x_dim=x_dim, e_dim=e_dim, hidden=128, layers=3, out_dim=256)
    model.eval()

    # Encode all reactions
    embeddings = []
    with torch.no_grad():
        for data in graphs:
            z = model(data)
            embeddings.append(z)

    embeddings = torch.cat(embeddings, dim=0)

    # Compute pairwise similarities
    print("\nPairwise Cosine Similarities:")
    similarity_matrix = embeddings @ embeddings.t()

    for i in range(len(reactions)):
        for j in range(i + 1, len(reactions)):
            sim = similarity_matrix[i, j].item()
            print(f"  Reaction {i+1} vs {j+1}: {sim:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        demo_single_reaction()
        demo_multiple_reactions()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
