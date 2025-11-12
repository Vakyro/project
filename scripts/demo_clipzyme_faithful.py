"""
CLIPZyme faithful implementation demonstration.

This script demonstrates the complete pipeline with architectures
matching the CLIPZyme paper exactly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import csv

from protein_encoder.egnn import ProteinEncoderEGNN
from reaction_encoder.dmpnn import ReactionDMPNN
from reaction_encoder.features_clipzyme import reaction_to_graphs_clipzyme
from reaction_encoder.loss import clip_loss
from torch_geometric.data import Data


def generate_dummy_structure(sequence, method='extended'):
    """
    Generate dummy 3D structure coordinates for demonstration.

    In production, use AlphaFold-predicted structures.

    Args:
        sequence: Amino acid sequence
        method: 'extended' or 'helix'

    Returns:
        Cα coordinates (N, 3)
    """
    n_residues = len(sequence)

    if method == 'extended':
        coords = torch.zeros(n_residues, 3)
        coords[:, 0] = torch.arange(n_residues) * 3.8
        coords += torch.randn(n_residues, 3) * 0.5
    elif method == 'helix':
        coords = torch.zeros(n_residues, 3)
        for i in range(n_residues):
            angle = (i * 2 * np.pi) / 3.6
            coords[i, 0] = 2.3 * np.cos(angle)
            coords[i, 1] = 2.3 * np.sin(angle)
            coords[i, 2] = i * 1.5

    return coords


def demo_faithful_clipzyme():
    """Demonstrate the complete faithful CLIPZyme implementation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nCLIPZyme Faithful Implementation Demo")
    print(f"Device: {device}\n")

    # Initialize models
    print("Initializing models...")
    protein_encoder = ProteinEncoderEGNN(
        plm_name="facebook/esm2_t33_650M_UR50D",
        hidden_dim=1280,
        num_layers=6,
        proj_dim=512,
        dropout=0.1,
        k_neighbors=30,
        distance_cutoff=10.0
    ).to(device).eval()

    reaction_encoder = ReactionDMPNN(
        node_dim=9,
        edge_dim=3,
        hidden_dim=1280,
        num_layers=5,
        proj_dim=512,
        dropout=0.1
    ).to(device).eval()

    print(f"Protein encoder: {sum(p.numel() for p in protein_encoder.parameters()):,} parameters")
    print(f"Reaction encoder: {sum(p.numel() for p in reaction_encoder.parameters()):,} parameters")

    # Load demo data from CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'enzyme_reactions.csv')
    enzyme_reactions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            enzyme_reactions.append({
                'name': row['name'],
                'sequence': row['sequence'],
                'reaction': row['reaction'],
                'description': row['description']
            })
            # Only load first 3 for this demo
            if len(enzyme_reactions) >= 3:
                break

    print(f"\nLoaded {len(enzyme_reactions)} enzyme-reaction pairs from CSV")

    # Encode proteins
    print("\nEncoding proteins...")
    sequences = [pair['sequence'] for pair in enzyme_reactions]
    protein_tokens = protein_encoder.tokenize(sequences, max_len=650)
    protein_tokens = {k: v.to(device) for k, v in protein_tokens.items()}

    coords_list = [generate_dummy_structure(seq, method='helix') for seq in sequences]

    with torch.no_grad():
        protein_embeddings = protein_encoder(protein_tokens, coords_list)

    print(f"Protein embeddings: {protein_embeddings.shape}")

    # Encode reactions
    print("\nEncoding reactions...")
    reaction_embeddings_list = []

    for i, pair in enumerate(enzyme_reactions):
        try:
            rxn_data = reaction_to_graphs_clipzyme(pair['reaction'])

            substrate_data = Data(
                x=rxn_data['substrate']['x'].to(device),
                edge_index=rxn_data['substrate']['edge_index'].to(device),
                edge_attr=rxn_data['substrate']['edge_attr'].to(device)
            )

            product_data = Data(
                x=rxn_data['product']['x'].to(device),
                edge_index=rxn_data['product']['edge_index'].to(device),
                edge_attr=rxn_data['product']['edge_attr'].to(device)
            )

            with torch.no_grad():
                z = reaction_encoder(substrate_data, product_data, rxn_data['atom_mapping'])

            reaction_embeddings_list.append(z)

        except Exception as e:
            print(f"Warning: Failed to encode reaction {i+1}: {e}")
            z = torch.zeros(1, 512, device=device)
            reaction_embeddings_list.append(z)

    reaction_embeddings = torch.cat(reaction_embeddings_list, dim=0)
    print(f"Reaction embeddings: {reaction_embeddings.shape}")

    # Compute similarities
    print("\nComputing similarities...")
    similarity = protein_embeddings @ reaction_embeddings.t()

    print("\nSimilarity Matrix:")
    print("                              R1      R2      R3")
    for i, pair in enumerate(enzyme_reactions):
        name = pair['name'][:28]
        sims = [f"{similarity[i, j].item():.3f}" for j in range(len(enzyme_reactions))]
        print(f"{name:28s}  " + "  ".join(sims))

    # CLIP loss
    loss = clip_loss(protein_embeddings, reaction_embeddings, temperature=0.07)
    print(f"\nCLIP Loss: {loss.item():.4f}")

    print("\nModel Architecture Summary:")
    print("  Protein: EGNN (6 layers, hidden=1280)")
    print("  Reaction: Two-Stage DMPNN (5 layers, hidden=1280)")
    print("  Features: 9 node, 3 edge (CLIPZyme standard)")
    print("  Total parameters: ~770M")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    try:
        demo_faithful_clipzyme()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
