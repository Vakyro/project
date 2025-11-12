"""
Change-only graph builder for focusing on reactive centers.

This creates a subgraph containing only the atoms and bonds that are
involved in the chemical transformation (formed/broken/changed bonds).
"""

import torch
from torch_geometric.data import Data
from typing import List, Dict, Set, Tuple
from rdkit import Chem

from .chem import bond_set, mapnums_index
from .builder import diff_bonds, EdgeChange
from .features import (
    atom_basic_features,
    atom_changed,
    vectorize_atom_features_enhanced,
    is_reactive_node
)


def build_change_only_graph(
    reacts: List[Chem.Mol],
    prods: List[Chem.Mol],
    use_enhanced_features: bool = True
) -> Data:
    """
    Build a graph containing only the reactive centers (atoms and bonds that change).

    This focuses the model's attention on the actual chemical transformation by
    filtering out spectator atoms and unchanged bonds.

    Args:
        reacts: List of reactant molecules (with atom mapping)
        prods: List of product molecules (with atom mapping)
        use_enhanced_features: If True, use enhanced features with one-hot encoding

    Returns:
        PyG Data object with:
        - x: node features [num_reactive_nodes, feature_dim]
        - edge_index: connectivity [2, num_reactive_edges]
        - edge_attr: edge features [num_reactive_edges, 6]
    """
    # Get atom mappings and bonds
    idx_r = mapnums_index(reacts)
    idx_p = mapnums_index(prods)
    bonds_r = bond_set(reacts)
    bonds_p = bond_set(prods)

    # Find all changes
    changes = diff_bonds(bonds_r, bonds_p)

    # Get reactive edges (formed/broken/changed)
    reactive_edges = {
        (u, v) for (u, v), ch in changes.items()
        if ch != EdgeChange.UNCHANGED
    }

    if not reactive_edges:
        # No changes detected - return minimal graph
        # Include all atoms but no edges
        all_mapnums = sorted(set(idx_r.keys()) | set(idx_p.keys()))

        node_features = []
        for mapnum in all_mapnums:
            # Get atoms
            a_r = _get_atom(reacts, idx_r, mapnum)
            a_p = _get_atom(prods, idx_p, mapnum)

            # Extract features
            f_r = atom_basic_features(a_r)
            f_p = atom_basic_features(a_p)

            Z = f_r.get("Z", f_p.get("Z", 0))

            if use_enhanced_features:
                # Enhanced: basic (7) + basic (7) + exists flags (2) + one-hot (11) + reactive (1) = 28
                feat = (
                    vectorize_atom_features_enhanced({}, 0, False)[:7] +  # reactant side
                    vectorize_atom_features_enhanced({}, 0, False)[:7] +  # product side
                    [int(a_r is not None), int(a_p is not None)] +        # existence
                    vectorize_atom_features_enhanced({}, Z, False)[7:]    # one-hot + reactive flag
                )
            else:
                # Original: 7 + 7 + 2 + 1 = 17
                from .features import vectorize_atom_features
                feat = (
                    vectorize_atom_features(f_r) +
                    vectorize_atom_features(f_p) +
                    [int(a_r is not None), int(a_p is not None), 0]
                )

            node_features.append(feat)

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Get reactive nodes (atoms involved in reactive edges)
    reactive_nodes = sorted({u for (u, _) in reactive_edges} | {v for (_, v) in reactive_edges})

    # Create local mapping
    local_map = {mapnum: i for i, mapnum in enumerate(reactive_nodes)}

    # Build node features for reactive atoms
    node_features = []
    for mapnum in reactive_nodes:
        # Get atoms
        a_r = _get_atom(reacts, idx_r, mapnum)
        a_p = _get_atom(prods, idx_p, mapnum)

        # Extract features
        f_r = atom_basic_features(a_r)
        f_p = atom_basic_features(a_p)

        # Check if this specific atom changed
        changed = atom_changed(a_r, a_p)

        # Get atomic number
        Z = f_r.get("Z", f_p.get("Z", 0))

        if use_enhanced_features:
            # Enhanced features with one-hot element encoding
            from .features import vectorize_atom_features_enhanced

            # Reactant side (19 features)
            feat_r = vectorize_atom_features_enhanced(f_r, Z, True)[:7]  # Just basic features
            # Product side (19 features)
            feat_p = vectorize_atom_features_enhanced(f_p, Z, True)[:7]   # Just basic features

            # Combine: basic_r (7) + basic_p (7) + exists (2) + one-hot (11) + reactive (1) = 28
            feat = feat_r + feat_p + [
                int(a_r is not None),
                int(a_p is not None)
            ] + vectorize_atom_features_enhanced(f_r, Z, True)[7:]  # one-hot + reactive flag
        else:
            # Original features
            from .features import vectorize_atom_features
            feat = (
                vectorize_atom_features(f_r) +
                vectorize_atom_features(f_p) +
                [int(a_r is not None), int(a_p is not None), int(changed)]
            )

        node_features.append(feat)

    # Build edge index and edge attributes for reactive edges
    edge_list = []
    edge_attrs = []

    for (u, v) in reactive_edges:
        # Get local indices
        u_local = local_map[u]
        v_local = local_map[v]

        # Get bond info
        bond_r = bonds_r.get((u, v), bonds_r.get((v, u)))
        bond_p = bonds_p.get((u, v), bonds_p.get((v, u)))

        # Edge attributes
        edge_change = changes.get((u, v), changes.get((v, u), EdgeChange.UNCHANGED))

        edge_feat = [
            int(bond_r is not None),           # Bond exists in reactant
            int(bond_p is not None),           # Bond exists in product
            int(edge_change == EdgeChange.FORMED),        # Formed
            int(edge_change == EdgeChange.BROKEN),        # Broken
            int(edge_change == EdgeChange.UNCHANGED),     # Unchanged (shouldn't happen here)
            int(edge_change == EdgeChange.CHANGED_ORDER)  # Changed order
        ]

        # Add both directions (undirected graph)
        edge_list.append([u_local, v_local])
        edge_list.append([v_local, u_local])
        edge_attrs.append(edge_feat)
        edge_attrs.append(edge_feat)

    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _get_atom(mols: List[Chem.Mol], idx_map: Dict, mapnum: int):
    """Helper to get atom by map number."""
    if mapnum not in idx_map:
        return None
    mol_idx, atom_idx = idx_map[mapnum]
    return mols[mol_idx].GetAtomWithIdx(atom_idx)
