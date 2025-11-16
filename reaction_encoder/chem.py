"""
Parsing and chemistry utilities using RDKit.
Handles reaction SMILES parsing and atom mapping.
"""

from rdkit import Chem
from typing import Dict, Tuple, List


def parse_reaction_smiles(reaction_smiles: str):
    """
    Parse reaction SMILES string into reactant and product molecule objects.

    This function splits an atom-mapped reaction SMILES string into its constituent
    reactant and product molecules, validating that all components can be parsed correctly.

    Args:
        reaction_smiles: Reaction SMILES string with atom mapping
                        Format: "reactants>>products"
                        Example: "[N:1]=[N:2]>>[N:1]-[N:2]"
                        Multi-component reactions use '.' separator: "A.B>>C.D"

    Returns:
        Tuple containing:
            - reactant_molecules: List of RDKit Mol objects for reactants
            - product_molecules: List of RDKit Mol objects for products

    Raises:
        ValueError: If SMILES string is missing '>>' separator or contains invalid SMILES

    Example:
        >>> reactants, products = parse_reaction_smiles("[CH3:1][OH:2]>>[CH3:1][O:2][CH3:1]")
        >>> len(reactants), len(products)
        (1, 1)
    """
    # Check for required reaction arrow separator
    if ">>" not in reaction_smiles:
        raise ValueError(f"Invalid reaction SMILES (missing '>>' separator): {reaction_smiles}")

    # Split into reactants and products
    reactants_string, products_string = reaction_smiles.split(">>")

    # Split multi-component reactants/products (separated by '.')
    reactant_smiles_list = reactants_string.split(".") if reactants_string else []
    product_smiles_list = products_string.split(".") if products_string else []

    # Parse SMILES strings into RDKit molecule objects
    reactant_molecules = [Chem.MolFromSmiles(smiles, sanitize=True) for smiles in reactant_smiles_list]
    product_molecules = [Chem.MolFromSmiles(smiles, sanitize=True) for smiles in product_smiles_list]

    # Validate that all molecules were parsed successfully (None indicates parse failure)
    all_molecules = reactant_molecules + product_molecules
    for molecule in all_molecules:
        if molecule is None:
            raise ValueError(f"Failed to parse SMILES in reaction: {reaction_smiles}")

    return reactant_molecules, product_molecules


def mapnums_index(molecules: List[Chem.Mol]) -> Dict[int, Tuple[int, int]]:
    """
    Create lookup index mapping atom map numbers to their positions in molecule list.

    In atom-mapped reactions, each atom is assigned a unique map number (e.g., [C:1], [O:2]).
    This function creates a dictionary that maps these map numbers to the molecule and atom
    indices where they occur, enabling efficient lookup of atoms across multiple molecules.

    Args:
        molecules: List of RDKit Mol objects with atom map numbers assigned

    Returns:
        Dictionary mapping atom_map_number -> (molecule_index, atom_index)
        Only includes atoms with map numbers > 0

    Example:
        >>> from rdkit import Chem
        >>> mol1 = Chem.MolFromSmiles("[CH3:1][OH:2]")
        >>> mol2 = Chem.MolFromSmiles("[CH3:3][NH2:4]")
        >>> index = mapnums_index([mol1, mol2])
        >>> index[1]  # Carbon from first molecule
        (0, 0)
        >>> index[4]  # Nitrogen from second molecule
        (1, 1)
    """
    atom_map_index = {}

    # Iterate through all molecules and their atoms
    for molecule_idx, molecule in enumerate(molecules):
        for atom_idx, atom in enumerate(molecule.GetAtoms()):
            atom_map_number = atom.GetAtomMapNum()

            # Only index atoms that have a map number assigned
            if atom_map_number > 0:
                atom_map_index[atom_map_number] = (molecule_idx, atom_idx)

    return atom_map_index


def bond_set(molecules: List[Chem.Mol]) -> Dict[Tuple[int, int], Dict]:
    """
    Extract all bonds between atom-mapped atoms with their chemical properties.

    This function identifies bonds that connect atoms with map numbers and records
    their chemical properties. Only bonds between mapped atoms are included, as
    these are the atoms that participate in the chemical transformation.

    Args:
        molecules: List of RDKit Mol objects with atom-mapped atoms

    Returns:
        Dictionary mapping (atom_map_u, atom_map_v) -> bond_properties
        where atom_map_u < atom_map_v (canonically sorted), and bond_properties contains:
            - order: Bond order as integer (1=single, 2=double, 3=triple, 1.5=aromatic)
            - aromatic: Boolean indicating if bond is aromatic
            - conj: Boolean indicating if bond is conjugated
            - ring: Boolean indicating if bond is part of a ring system

    Note:
        - Only bonds between two mapped atoms (map number > 0) are included
        - Bond keys are sorted to ensure canonical representation: (1,2) not (2,1)
        - If a bond appears in multiple molecules, the last occurrence overwrites

    Example:
        >>> mol = Chem.MolFromSmiles("[C:1]=[C:2]-[O:3]")
        >>> bonds = bond_set([mol])
        >>> bonds[(1, 2)]  # Double bond between C1 and C2
        {'order': 2, 'aromatic': False, 'conj': True, 'ring': False}
    """
    bond_dictionary = {}

    # Iterate through all molecules
    for molecule in molecules:
        # Process each bond in the molecule
        for bond in molecule.GetBonds():
            # Get the two atoms connected by this bond
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            # Get their atom map numbers
            begin_map_num = begin_atom.GetAtomMapNum()
            end_map_num = end_atom.GetAtomMapNum()

            # Only process bonds between mapped atoms (both map numbers must be non-zero)
            if begin_map_num and end_map_num:
                # Create canonical bond key (smaller map number first)
                bond_key = tuple(sorted((begin_map_num, end_map_num)))

                # Store bond properties
                bond_dictionary[bond_key] = {
                    "order": int(bond.GetBondTypeAsDouble()),
                    "aromatic": bond.GetIsAromatic(),
                    "conj": bond.GetIsConjugated(),
                    "ring": bond.IsInRing(),
                }

    return bond_dictionary
