"""
Build transition state graphs from reaction SMILES.
Combines reactants and products into a unified graph representation.
"""

import torch
from torch_geometric.data import Data
from enum import IntEnum
from typing import List, Dict, Tuple

from .chem import mapnums_index, bond_set
from .features import (
    atom_basic_features,
    atom_changed,
    vectorize_atom_features,
    vectorize_atom_features_enhanced,
    is_reactive_node
)


class EdgeChange(IntEnum):
    """Enumeration of edge change types."""
    UNCHANGED = 0
    FORMED = 1
    BROKEN = 2
    CHANGED_ORDER = 3


def diff_bonds(react_bonds: Dict[Tuple[int, int], Dict],
               prod_bonds: Dict[Tuple[int, int], Dict]) -> Dict[Tuple[int, int], EdgeChange]:
    """
    Compare bonds between reactants and products to determine changes.

    Args:
        react_bonds: Bond dictionary from reactants
        prod_bonds: Bond dictionary from products

    Returns:
        Dictionary mapping edge tuples to EdgeChange enum values
    """
    all_keys = set(react_bonds) | set(prod_bonds)
    changes = {}

    for e in all_keys:
        r = react_bonds.get(e)
        p = prod_bonds.get(e)

        if r and p:
            # Bond exists in both - check if it changed
            if r["order"] == p["order"] and r["aromatic"] == p["aromatic"]:
                changes[e] = EdgeChange.UNCHANGED
            else:
                changes[e] = EdgeChange.CHANGED_ORDER
        elif p and not r:
            changes[e] = EdgeChange.FORMED
        elif r and not p:
            changes[e] = EdgeChange.BROKEN

    return changes


def build_transition_graph(reacts: List, prods: List, use_enhanced_features: bool = False) -> Data:
    """
    Build a PyTorch Geometric Data object representing the reaction transition state.

    The graph combines information from both reactants and products:
    - Nodes represent atoms (by map number)
    - Node features include properties from both reactant and product sides
    - Edges represent bonds with labels indicating changes (formed/broken/unchanged)

    Args:
        reacts: List of RDKit Mol objects (reactants)
        prods: List of RDKit Mol objects (products)
        use_enhanced_features: If True, use enhanced features with one-hot element encoding
                               and reactive node flags (28 features vs 16 features)

    Returns:
        PyTorch Geometric Data object with:
        - x: node features [num_nodes, feature_dim]
            - Original: 7 (reactant) + 7 (product) + 2 (existence) + 1 (changed) = 17
            - Enhanced: 7 (reactant) + 7 (product) + 2 (existence) + 11 (element one-hot) + 1 (reactive) = 28
        - edge_index: edge connectivity [2, num_edges]
        - edge_attr: edge features [num_edges, 6]
    """
    # Build atom mapping indices
    idx_r = mapnums_index(reacts)
    idx_p = mapnums_index(prods)

    # Get universe of mapped atoms
    mapnums = sorted(set(idx_r) | set(idx_p))
    mapnum_to_node = {m: i for i, m in enumerate(mapnums)}

    # Get bond sets and compute changes
    bonds_r = bond_set(reacts)
    bonds_p = bond_set(prods)
    changes = diff_bonds(bonds_r, bonds_p)

    # Find reactive edges for enhanced features
    reactive_edges = set()
    if use_enhanced_features:
        for e, ch in changes.items():
            if ch != EdgeChange.UNCHANGED:
                reactive_edges.add(e)

    # Build node features
    x_list = []

    for m in mapnums:
        ar = ap = None

        # Get atom from reactant side if it exists
        if m in idx_r:
            mi, ai = idx_r[m]
            ar = reacts[mi].GetAtomWithIdx(ai)

        # Get atom from product side if it exists
        if m in idx_p:
            mi, ai = idx_p[m]
            ap = prods[mi].GetAtomWithIdx(ai)

        # Extract features from both sides
        fr = atom_basic_features(ar) if ar else {}
        fp = atom_basic_features(ap) if ap else {}

        if use_enhanced_features:
            # Enhanced features with one-hot element encoding
            Z = fr.get("Z", fp.get("Z", 0))
            is_reactive = is_reactive_node(m, reactive_edges)

            # Reactant basic (7) + Product basic (7) + Existence (2) + One-hot (11) + Reactive (1) = 28
            xr = vectorize_atom_features(fr)  # 7 features
            xp = vectorize_atom_features(fp)  # 7 features
            exists_flags = [int(ar is not None), int(ap is not None)]  # 2 features

            # Get one-hot and reactive flag (12 features)
            from .features import one_hot_element
            element_oh = one_hot_element(Z)  # 11 features
            reactive_flag = [int(is_reactive)]  # 1 feature

            x = torch.tensor(xr + xp + exists_flags + element_oh + reactive_flag, dtype=torch.float32)
        else:
            # Original features
            xr = vectorize_atom_features(fr)
            xp = vectorize_atom_features(fp)

            # Add existence flags
            exists_r = [1 if ar else 0]
            exists_p = [1 if ap else 0]

            # Mark if atom changed
            changed = atom_changed(ar, ap)
            changed_flag = [1.0 if changed else 0.0]

            # Combine: 7 + 7 + 2 + 1 = 17
            x = torch.tensor(xr + xp + exists_r + exists_p + changed_flag, dtype=torch.float32)

        x_list.append(x)

    # Stack node features
    x = torch.stack(x_list, dim=0)

    # Build edges and edge attributes
    edge_index = []
    edge_attr = []

    for e, ch in changes.items():
        u, v = e
        iu, iv = mapnum_to_node[u], mapnum_to_node[v]

        # Add undirected edges (both directions)
        edge_index += [[iu, iv], [iv, iu]]

        # Edge attributes encode:
        # - Bond exists in reactant
        # - Bond exists in product
        # - Bond formed
        # - Bond broken
        # - Bond unchanged
        # - Bond changed order
        r = bonds_r.get(e) is not None
        p = bonds_p.get(e) is not None
        formed = int(ch == EdgeChange.FORMED)
        broken = int(ch == EdgeChange.BROKEN)
        unchg = int(ch == EdgeChange.UNCHANGED)
        chgord = int(ch == EdgeChange.CHANGED_ORDER)

        attr = [int(r), int(p), formed, broken, unchg, chgord]
        edge_attr += [attr, attr]  # Duplicate for both directions

    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.zeros((0, 6), dtype=torch.float32),
    )
    data.num_nodes = x.size(0)

    return data
