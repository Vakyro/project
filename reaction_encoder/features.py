"""
Atom and bond featurization utilities.
"""

from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from typing import Dict, Optional, Set, Tuple

# Common elements in organic chemistry
COMMON_ELEMENTS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I


def atom_basic_features(atom: Optional[Chem.Atom]) -> Dict:
    """
    Extract basic features from an atom.

    Args:
        atom: RDKit Atom object or None

    Returns:
        Dictionary with atom features:
        - Z: atomic number
        - degree: number of directly bonded neighbors
        - formal_charge: formal charge
        - is_aromatic: aromaticity flag
        - hyb: hybridization type (as integer)
        - num_h: total number of hydrogens
        - in_ring: whether atom is in a ring
    """
    if atom is None:
        return {}

    return {
        "Z": atom.GetAtomicNum(),
        "degree": atom.GetDegree(),
        "formal_charge": atom.GetFormalCharge(),
        "is_aromatic": atom.GetIsAromatic(),
        "hyb": int(atom.GetHybridization()) if atom.GetHybridization() != HybridizationType.UNSPECIFIED else 0,
        "num_h": atom.GetTotalNumHs(),
        "in_ring": atom.IsInRing(),
    }


def atom_changed(a_react: Optional[Chem.Atom], a_prod: Optional[Chem.Atom]) -> bool:
    """
    Determine if an atom has changed between reactant and product.

    An atom is considered "changed" (reactive) if:
    - It only exists on one side (added/removed)
    - Its formal charge, hybridization, number of hydrogens, or aromaticity changed

    Args:
        a_react: Atom in reactant (or None if doesn't exist)
        a_prod: Atom in product (or None if doesn't exist)

    Returns:
        True if atom changed, False otherwise
    """
    if a_react is None or a_prod is None:
        return True  # Added or removed

    f_r = atom_basic_features(a_react)
    f_p = atom_basic_features(a_prod)

    # Check key chemical properties that indicate reactivity
    keys = ["formal_charge", "hyb", "num_h", "is_aromatic"]
    return any(f_r[k] != f_p[k] for k in keys)


def one_hot_element(Z: int) -> list:
    """
    Create one-hot encoding for element type.

    Args:
        Z: Atomic number

    Returns:
        List of length 11: one-hot for common elements + 'other' category
        Categories: H, C, N, O, F, P, S, Cl, Br, I, Other
    """
    vec = [1 if Z == z else 0 for z in COMMON_ELEMENTS]
    # Add "other" category for rare elements
    vec.append(1 if Z not in COMMON_ELEMENTS and Z > 0 else 0)
    return vec


def is_reactive_node(mapnum: int, reactive_edges: Set[Tuple[int, int]]) -> bool:
    """
    Determine if a node (atom) is reactive based on its participation in reactive edges.

    Args:
        mapnum: Atom map number
        reactive_edges: Set of (u, v) tuples representing bonds that formed/broke/changed

    Returns:
        True if atom participates in any reactive edge
    """
    for u, v in reactive_edges:
        if u == mapnum or v == mapnum:
            return True
    return False


def vectorize_atom_features(feat: Dict) -> list:
    """
    Convert atom feature dictionary to a flat list for tensor construction.

    Args:
        feat: Dictionary from atom_basic_features

    Returns:
        List of numeric feature values (7 features)
    """
    return [
        feat.get("Z", 0),
        feat.get("degree", 0),
        feat.get("formal_charge", 0),
        int(feat.get("is_aromatic", 0)),
        feat.get("hyb", 0),
        feat.get("num_h", 0),
        int(feat.get("in_ring", 0))
    ]


def vectorize_atom_features_enhanced(feat: Dict, Z: int, is_reactive: bool) -> list:
    """
    Convert atom features to enhanced vector with one-hot element encoding.

    Args:
        feat: Dictionary from atom_basic_features
        Z: Atomic number for one-hot encoding
        is_reactive: Whether this atom is involved in reactive bonds

    Returns:
        List with: [basic features (7)] + [one-hot element (11)] + [is_reactive (1)] = 19 features
    """
    basic = vectorize_atom_features(feat)
    element_oh = one_hot_element(Z)
    reactive_flag = [int(is_reactive)]
    return basic + element_oh + reactive_flag
