"""
Build transition state graphs from reaction SMILES.

This module constructs graph representations of chemical reactions by combining
information from reactant and product molecules into a unified transition state graph.
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
    """
    Enumeration of bond change types in chemical reactions.

    Classifies how bonds change during a reaction:
        - UNCHANGED: Bond exists in both reactants and products with same properties
        - FORMED: Bond appears in products but not in reactants (bond formation)
        - BROKEN: Bond exists in reactants but not in products (bond breaking)
        - CHANGED_ORDER: Bond exists in both but with different order (e.g., single to double)
    """
    UNCHANGED = 0
    FORMED = 1
    BROKEN = 2
    CHANGED_ORDER = 3


def diff_bonds(
    reactant_bonds: Dict[Tuple[int, int], Dict],
    product_bonds: Dict[Tuple[int, int], Dict]
) -> Dict[Tuple[int, int], EdgeChange]:
    """
    Compare bonds between reactants and products to identify bond changes.

    This function analyzes which bonds are formed, broken, or modified during
    a chemical reaction by comparing the bond dictionaries from reactants and products.

    Args:
        reactant_bonds: Bond dictionary from reactant molecules
                       Format: {(atom_map_1, atom_map_2): bond_properties}
        product_bonds: Bond dictionary from product molecules
                      Format: {(atom_map_1, atom_map_2): bond_properties}

    Returns:
        Dictionary mapping bond (atom_map_1, atom_map_2) to EdgeChange enum value
        indicating how each bond changed during the reaction

    Example:
        >>> reactant_bonds = {(1, 2): {'order': 2, 'aromatic': False}}
        >>> product_bonds = {(1, 2): {'order': 1, 'aromatic': False}, (2, 3): {'order': 1, 'aromatic': False}}
        >>> changes = diff_bonds(reactant_bonds, product_bonds)
        >>> changes[(1, 2)]  # Bond order changed
        <EdgeChange.CHANGED_ORDER: 3>
        >>> changes[(2, 3)]  # New bond formed
        <EdgeChange.FORMED: 1>
    """
    # Get all unique bonds from both reactants and products
    all_bond_keys = set(reactant_bonds) | set(product_bonds)
    bond_changes = {}

    # Classify each bond
    for bond_key in all_bond_keys:
        reactant_bond = reactant_bonds.get(bond_key)
        product_bond = product_bonds.get(bond_key)

        if reactant_bond and product_bond:
            # Bond exists in both - check if properties changed
            if (reactant_bond["order"] == product_bond["order"] and
                reactant_bond["aromatic"] == product_bond["aromatic"]):
                bond_changes[bond_key] = EdgeChange.UNCHANGED
            else:
                bond_changes[bond_key] = EdgeChange.CHANGED_ORDER
        elif product_bond and not reactant_bond:
            # Bond only in products - it was formed
            bond_changes[bond_key] = EdgeChange.FORMED
        elif reactant_bond and not product_bond:
            # Bond only in reactants - it was broken
            bond_changes[bond_key] = EdgeChange.BROKEN

    return bond_changes


def build_transition_graph(
    reactant_molecules: List,
    product_molecules: List,
    use_enhanced_features: bool = False
) -> Data:
    """
    Build a PyTorch Geometric graph representing the reaction transition state.

    This function creates a unified graph representation that captures the transformation
    from reactants to products. Each node represents an atom (identified by map number),
    and edges represent bonds. Node features encode properties from both reactant and
    product states, while edge features indicate bond changes.

    The transition state graph is central to learning reaction representations:
    - It explicitly models which atoms and bonds participate in the reaction
    - Node features capture how atoms change during transformation
    - Edge labels classify bonds as formed, broken, or modified

    Args:
        reactant_molecules: List of RDKit Mol objects representing reactants
        product_molecules: List of RDKit Mol objects representing products
        use_enhanced_features: If True, use 28-dimensional node features including
                              one-hot element encoding and reactive node flags.
                              If False, use 17-dimensional basic features.

    Returns:
        PyTorch Geometric Data object containing:
            - x: Node feature matrix, shape [num_nodes, feature_dim]
              * Basic (17D): reactant_features (7) + product_features (7) +
                            exists_in_reactant (1) + exists_in_product (1) + changed (1)
              * Enhanced (28D): reactant_features (7) + product_features (7) +
                               exists_flags (2) + element_one_hot (11) + is_reactive (1)
            - edge_index: Edge connectivity in COO format, shape [2, num_edges]
            - edge_attr: Edge feature matrix, shape [num_edges, 6]
              Features: [exists_reactant, exists_product, formed, broken, unchanged, changed_order]

    Example:
        >>> from rdkit import Chem
        >>> reactants = [Chem.MolFromSmiles("[C:1]=[O:2]")]
        >>> products = [Chem.MolFromSmiles("[C:1]-[O:2]")]
        >>> graph = build_transition_graph(reactants, products)
        >>> graph.x.shape  # Node features
        torch.Size([2, 17])
        >>> graph.edge_index.shape  # Edges
        torch.Size([2, 2])  # Bidirectional edge
    """
    # Build lookup indices: atom_map_number -> (molecule_idx, atom_idx)
    reactant_atom_index = mapnums_index(reactant_molecules)
    product_atom_index = mapnums_index(product_molecules)

    # Get all unique atom map numbers from both reactants and products
    all_atom_map_numbers = sorted(set(reactant_atom_index) | set(product_atom_index))

    # Create mapping from atom_map_number to graph node index
    atom_map_to_node_index = {
        map_num: node_idx
        for node_idx, map_num in enumerate(all_atom_map_numbers)
    }

    # Extract bond sets from reactants and products
    reactant_bonds = bond_set(reactant_molecules)
    product_bonds = bond_set(product_molecules)

    # Determine how bonds change during the reaction
    bond_changes = diff_bonds(reactant_bonds, product_bonds)

    # Identify which bonds participate in the reaction (for enhanced features)
    reactive_bond_set = set()
    if use_enhanced_features:
        for bond_key, change_type in bond_changes.items():
            if change_type != EdgeChange.UNCHANGED:
                # This bond is formed, broken, or changes order
                reactive_bond_set.add(bond_key)

    # Build node feature vectors for each atom
    node_feature_list = []

    for atom_map_number in all_atom_map_numbers:
        reactant_atom = None
        product_atom = None

        # Get atom from reactant side if it exists there
        if atom_map_number in reactant_atom_index:
            molecule_idx, atom_idx = reactant_atom_index[atom_map_number]
            reactant_atom = reactant_molecules[molecule_idx].GetAtomWithIdx(atom_idx)

        # Get atom from product side if it exists there
        if atom_map_number in product_atom_index:
            molecule_idx, atom_idx = product_atom_index[atom_map_number]
            product_atom = product_molecules[molecule_idx].GetAtomWithIdx(atom_idx)

        # Extract chemical features from reactant and product atoms
        reactant_features = atom_basic_features(reactant_atom) if reactant_atom else {}
        product_features = atom_basic_features(product_atom) if product_atom else {}

        if use_enhanced_features:
            # Enhanced features: 28-dimensional
            # Get atomic number (Z) from whichever side exists
            atomic_number = reactant_features.get("Z", product_features.get("Z", 0))

            # Check if this atom participates in reactive bonds
            is_reactive = is_reactive_node(atom_map_number, reactive_bond_set)

            # Vectorize basic features from both sides (7 features each)
            reactant_vec = vectorize_atom_features(reactant_features)  # 7 features
            product_vec = vectorize_atom_features(product_features)    # 7 features

            # Existence flags (2 features)
            exists_flags = [
                int(reactant_atom is not None),
                int(product_atom is not None)
            ]

            # One-hot element encoding and reactive flag (12 features)
            from .features import one_hot_element
            element_one_hot = one_hot_element(atomic_number)  # 11 features
            reactive_flag = [int(is_reactive)]                # 1 feature

            # Concatenate all features: 7 + 7 + 2 + 11 + 1 = 28
            node_features = torch.tensor(
                reactant_vec + product_vec + exists_flags + element_one_hot + reactive_flag,
                dtype=torch.float32
            )
        else:
            # Basic features: 17-dimensional
            # Vectorize basic features from both sides (7 features each)
            reactant_vec = vectorize_atom_features(reactant_features)
            product_vec = vectorize_atom_features(product_features)

            # Existence flags (2 features total, split for clarity)
            exists_in_reactant = [1 if reactant_atom else 0]
            exists_in_product = [1 if product_atom else 0]

            # Check if atom properties changed between reactant and product
            atom_has_changed = atom_changed(reactant_atom, product_atom)
            change_flag = [1.0 if atom_has_changed else 0.0]

            # Concatenate all features: 7 + 7 + 1 + 1 + 1 = 17
            node_features = torch.tensor(
                reactant_vec + product_vec + exists_in_reactant + exists_in_product + change_flag,
                dtype=torch.float32
            )

        node_feature_list.append(node_features)

    # Stack all node features into matrix
    node_feature_matrix = torch.stack(node_feature_list, dim=0)

    # Build edges and edge attributes
    edge_index_list = []
    edge_attribute_list = []

    # Process each bond and create bidirectional edges
    for bond_key, change_type in bond_changes.items():
        # bond_key is (atom_map_u, atom_map_v) where u < v
        atom_map_u, atom_map_v = bond_key

        # Convert atom map numbers to node indices
        node_idx_u = atom_map_to_node_index[atom_map_u]
        node_idx_v = atom_map_to_node_index[atom_map_v]

        # Add bidirectional edges (undirected graph represented as directed)
        edge_index_list += [[node_idx_u, node_idx_v], [node_idx_v, node_idx_u]]

        # Build edge attribute vector (6 features):
        # 1. exists_in_reactant: 1 if bond exists in reactants, 0 otherwise
        # 2. exists_in_product: 1 if bond exists in products, 0 otherwise
        # 3. formed: 1 if bond was formed (only in products), 0 otherwise
        # 4. broken: 1 if bond was broken (only in reactants), 0 otherwise
        # 5. unchanged: 1 if bond unchanged, 0 otherwise
        # 6. changed_order: 1 if bond order changed, 0 otherwise

        exists_in_reactant = int(reactant_bonds.get(bond_key) is not None)
        exists_in_product = int(product_bonds.get(bond_key) is not None)
        bond_formed = int(change_type == EdgeChange.FORMED)
        bond_broken = int(change_type == EdgeChange.BROKEN)
        bond_unchanged = int(change_type == EdgeChange.UNCHANGED)
        bond_order_changed = int(change_type == EdgeChange.CHANGED_ORDER)

        edge_attributes = [
            exists_in_reactant,
            exists_in_product,
            bond_formed,
            bond_broken,
            bond_unchanged,
            bond_order_changed
        ]

        # Add attributes for both edge directions (same attributes for both)
        edge_attribute_list += [edge_attributes, edge_attributes]

    # Convert to PyTorch Geometric Data object
    transition_graph = Data(
        # Node feature matrix
        x=node_feature_matrix,

        # Edge connectivity in COO format [2, num_edges]
        edge_index=torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
                   if edge_index_list
                   else torch.zeros((2, 0), dtype=torch.long),

        # Edge feature matrix [num_edges, 6]
        edge_attr=torch.tensor(edge_attribute_list, dtype=torch.float32)
                  if edge_attribute_list
                  else torch.zeros((0, 6), dtype=torch.float32),
    )

    # Explicitly set number of nodes (required by some PyG operations)
    transition_graph.num_nodes = node_feature_matrix.size(0)

    return transition_graph
