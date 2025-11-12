"""
Atom and bond featurization matching CLIPZyme paper exactly.

CLIPZyme uses:
- 9 node features: atomic number, chirality, degree, formal charge,
                   num hydrogens, radical electrons, hybridization,
                   aromaticity, ring membership
- 3 edge features: bond type, stereochemistry, conjugation
"""

from rdkit import Chem
from rdkit.Chem import ChiralType, BondType, BondStereo
from typing import Optional


def get_atom_features_clipzyme(atom: Chem.Atom) -> list:
    """
    Extract 9 atom features as specified in CLIPZyme paper.

    Features:
    1. Atomic number (Z)
    2. Chirality (0=unspecified, 1=tetrahedral CW, 2=tetrahedral CCW, 3=other)
    3. Degree (number of bonded neighbors)
    4. Formal charge
    5. Number of hydrogens
    6. Number of radical electrons
    7. Hybridization (0=unspecified, 1=S, 2=SP, 3=SP2, 4=SP3, 5=SP3D, 6=SP3D2, 7=other)
    8. Aromaticity (0 or 1)
    9. Ring membership (0 or 1)

    Args:
        atom: RDKit Atom object

    Returns:
        List of 9 numeric features
    """
    # 1. Atomic number
    atomic_num = atom.GetAtomicNum()

    # 2. Chirality
    chiral_tag = atom.GetChiralTag()
    if chiral_tag == ChiralType.CHI_UNSPECIFIED:
        chirality = 0
    elif chiral_tag == ChiralType.CHI_TETRAHEDRAL_CW:
        chirality = 1
    elif chiral_tag == ChiralType.CHI_TETRAHEDRAL_CCW:
        chirality = 2
    else:
        chirality = 3  # Other types (square planar, etc.)

    # 3. Degree
    degree = atom.GetDegree()

    # 4. Formal charge
    formal_charge = atom.GetFormalCharge()

    # 5. Number of hydrogens
    num_hs = atom.GetTotalNumHs()

    # 6. Number of radical electrons
    num_radical = atom.GetNumRadicalElectrons()

    # 7. Hybridization
    hyb = atom.GetHybridization()
    hyb_type = {
        Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
        Chem.rdchem.HybridizationType.S: 1,
        Chem.rdchem.HybridizationType.SP: 2,
        Chem.rdchem.HybridizationType.SP2: 3,
        Chem.rdchem.HybridizationType.SP3: 4,
        Chem.rdchem.HybridizationType.SP3D: 5,
        Chem.rdchem.HybridizationType.SP3D2: 6,
    }.get(hyb, 7)  # 7 for other/unknown

    # 8. Aromaticity
    is_aromatic = int(atom.GetIsAromatic())

    # 9. Ring membership
    in_ring = int(atom.IsInRing())

    return [
        atomic_num,
        chirality,
        degree,
        formal_charge,
        num_hs,
        num_radical,
        hyb_type,
        is_aromatic,
        in_ring
    ]


def get_bond_features_clipzyme(bond: Chem.Bond) -> list:
    """
    Extract 3 bond features as specified in CLIPZyme paper.

    Features:
    1. Bond type (1=SINGLE, 2=DOUBLE, 3=TRIPLE, 4=AROMATIC)
    2. Stereochemistry (0=none, 1=E, 2=Z, 3=cis, 4=trans, 5=any)
    3. Conjugation (0 or 1)

    Args:
        bond: RDKit Bond object

    Returns:
        List of 3 numeric features
    """
    # 1. Bond type
    bond_type_map = {
        BondType.SINGLE: 1,
        BondType.DOUBLE: 2,
        BondType.TRIPLE: 3,
        BondType.AROMATIC: 4
    }
    bond_type = bond_type_map.get(bond.GetBondType(), 1)

    # 2. Stereochemistry
    stereo = bond.GetStereo()
    stereo_map = {
        BondStereo.STEREONONE: 0,
        BondStereo.STEREOE: 1,
        BondStereo.STEREOZ: 2,
        BondStereo.STEREOCIS: 3,
        BondStereo.STEREOTRANS: 4,
        BondStereo.STEREOANY: 5
    }
    stereo_val = stereo_map.get(stereo, 0)

    # 3. Conjugation
    is_conjugated = int(bond.GetIsConjugated())

    return [bond_type, stereo_val, is_conjugated]


def mol_to_graph_clipzyme(mol: Chem.Mol, use_atom_mapping: bool = True):
    """
    Convert RDKit molecule to graph representation with CLIPZyme features.

    Args:
        mol: RDKit molecule
        use_atom_mapping: If True, use atom map numbers as indices

    Returns:
        Dictionary with:
            - atom_features: List of atom feature vectors (N, 9)
            - bond_features: List of bond feature vectors (E, 3)
            - edge_index: List of [src, dst] pairs
            - atom_map: Dict mapping atom_map_num -> atom_idx (if use_atom_mapping)
    """
    import torch

    # Get atoms
    atoms = mol.GetAtoms()
    n_atoms = mol.GetNumAtoms()

    # Extract atom features
    atom_features = []
    atom_map = {}

    for atom in atoms:
        feat = get_atom_features_clipzyme(atom)
        atom_features.append(feat)

        if use_atom_mapping:
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                atom_map[map_num] = atom.GetIdx()

    # Extract bonds and their features
    edge_index = []
    bond_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        feat = get_bond_features_clipzyme(bond)

        # Add both directions for undirected graph
        edge_index.append([i, j])
        bond_features.append(feat)

        edge_index.append([j, i])
        bond_features.append(feat)

    # Convert to tensors
    if len(atom_features) > 0:
        atom_features = torch.tensor(atom_features, dtype=torch.float)
    else:
        atom_features = torch.zeros((0, 9), dtype=torch.float)

    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # (2, num_edges)
        bond_features = torch.tensor(bond_features, dtype=torch.float)  # (num_edges, 3)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        bond_features = torch.zeros((0, 3), dtype=torch.float)

    return {
        'x': atom_features,
        'edge_index': edge_index,
        'edge_attr': bond_features,
        'atom_map': atom_map
    }


def reaction_to_graphs_clipzyme(reaction_smiles: str):
    """
    Parse atom-mapped reaction SMILES and convert to graphs.

    Args:
        reaction_smiles: Reaction SMILES with atom mapping
                        Example: "[CH3:1][OH:2]>>[CH3:1][O:2][CH3:1]"

    Returns:
        Dictionary with:
            - substrate: Graph dict for substrate(s)
            - product: Graph dict for product(s)
            - atom_mapping: Dict mapping substrate atoms to product atoms
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Split reaction SMILES
    if '>>' not in reaction_smiles:
        raise ValueError(f"Invalid reaction SMILES (missing >>): {reaction_smiles}")

    parts = reaction_smiles.split('>>')
    if len(parts) != 2:
        raise ValueError(f"Invalid reaction SMILES format: {reaction_smiles}")

    reactant_smiles, product_smiles = parts

    # Parse reactants and products
    reactant_mols = [Chem.MolFromSmiles(s.strip()) for s in reactant_smiles.split('.') if s.strip()]
    product_mols = [Chem.MolFromSmiles(s.strip()) for s in product_smiles.split('.') if s.strip()]

    if not reactant_mols or not product_mols:
        raise ValueError(f"Failed to parse reaction: {reaction_smiles}")

    reactants = [m for m in reactant_mols if m is not None]
    products = [m for m in product_mols if m is not None]

    if not reactants or not products:
        raise ValueError(f"Failed to parse molecules in reaction: {reaction_smiles}")

    # For multi-component reactions, take the largest molecule (main substrate/product)
    # Ignore small molecules like H2O, cofactors, etc.
    def get_main_molecule(mols):
        """Get the largest molecule by atom count."""
        if len(mols) == 1:
            return mols[0]
        # Return molecule with most atoms
        return max(mols, key=lambda m: m.GetNumAtoms())

    substrate_mol = get_main_molecule(reactants)
    product_mol = get_main_molecule(products)

    # Convert to graphs
    substrate_graph = mol_to_graph_clipzyme(substrate_mol, use_atom_mapping=True)
    product_graph = mol_to_graph_clipzyme(product_mol, use_atom_mapping=True)

    # Build atom mapping between substrate and product
    # Map substrate atom indices to product atom indices
    atom_mapping = {}
    sub_map = substrate_graph['atom_map']  # {map_num: atom_idx}
    prod_map = product_graph['atom_map']   # {map_num: atom_idx}

    for map_num in sub_map:
        if map_num in prod_map:
            atom_mapping[sub_map[map_num]] = prod_map[map_num]

    return {
        'substrate': substrate_graph,
        'product': product_graph,
        'atom_mapping': atom_mapping
    }


def normalize_features(features, feature_type='atom'):
    """
    Normalize features (optional preprocessing).

    Args:
        features: Tensor of features (N, D)
        feature_type: 'atom' or 'bond'

    Returns:
        Normalized features
    """
    # CLIPZyme may or may not use normalization
    # This is a placeholder if needed
    return features
