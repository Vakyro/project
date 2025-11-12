"""
Parsing and chemistry utilities using RDKit.
Handles reaction SMILES parsing and atom mapping.
"""

from rdkit import Chem
from typing import Dict, Tuple, List


def parse_reaction_smiles(rxn_smiles: str):
    """
    Parse reaction SMILES into reactant and product molecules.

    Args:
        rxn_smiles: Reaction SMILES with atom mapping (e.g., "[N:1]=[N:2]>>[N:1]-[N:2]")

    Returns:
        Tuple of (reactant_mols, product_mols) where each is a list of RDKit Mol objects

    Raises:
        ValueError: If SMILES is invalid or not properly formatted
    """
    if ">>" not in rxn_smiles:
        raise ValueError(f"Invalid reaction SMILES (missing '>>'): {rxn_smiles}")

    reactants_str, products_str = rxn_smiles.split(">>")
    reactant_smiles = reactants_str.split(".") if reactants_str else []
    product_smiles = products_str.split(".") if products_str else []

    reacts = [Chem.MolFromSmiles(s, sanitize=True) for s in reactant_smiles]
    prods = [Chem.MolFromSmiles(s, sanitize=True) for s in product_smiles]

    # Validate all molecules were parsed successfully
    for m in reacts + prods:
        if m is None:
            raise ValueError(f"Bad SMILES in reaction: {rxn_smiles}")

    return reacts, prods


def mapnums_index(mols: List[Chem.Mol]) -> Dict[int, Tuple[int, int]]:
    """
    Create index mapping atom map numbers to (molecule_idx, atom_idx).

    Args:
        mols: List of RDKit Mol objects with atom map numbers

    Returns:
        Dictionary mapping mapnum -> (mol_idx, atom_idx)
    """
    idx = {}
    for mi, mol in enumerate(mols):
        for ai, atom in enumerate(mol.GetAtoms()):
            mapnum = atom.GetAtomMapNum()
            if mapnum > 0:
                idx[mapnum] = (mi, ai)
    return idx


def bond_set(mols: List[Chem.Mol]) -> Dict[Tuple[int, int], Dict]:
    """
    Extract all bonds between mapped atoms.

    Args:
        mols: List of RDKit Mol objects

    Returns:
        Dictionary with key (mapu, mapv) where u < v, and value containing bond properties:
        - order: bond order as integer
        - aromatic: whether bond is aromatic
        - conj: whether bond is conjugated
        - ring: whether bond is in a ring
    """
    edges = {}
    for mol in mols:
        for b in mol.GetBonds():
            a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
            m1, m2 = a1.GetAtomMapNum(), a2.GetAtomMapNum()
            if m1 and m2:  # Only consider bonds between mapped atoms
                u, v = sorted((m1, m2))
                edges[(u, v)] = {
                    "order": int(b.GetBondTypeAsDouble()),
                    "aromatic": b.GetIsAromatic(),
                    "conj": b.GetIsConjugated(),
                    "ring": b.IsInRing(),
                }
    return edges
