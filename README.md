# Reaction Encoder - CLIPZyme Style

A Graph Neural Network (GNN) based encoder for chemical reactions inspired by the [CLIPZyme](https://arxiv.org/abs/2402.06748) approach. This project encodes reactions as transition state graphs that capture the chemical transformation between reactants and products.

## Overview

This encoder represents chemical reactions by:
1. Parsing atom-mapped reaction SMILES
2. Building a unified "pseudo-transition state" graph combining reactant and product information
3. Labeling bonds as formed/broken/unchanged
4. Labeling atoms as reactive/unreactive
5. Processing the graph with a GNN to produce a fixed-size reaction embedding

This approach focuses on representing the **chemical transformation** rather than just the molecular structures.

## Features

- **Atom-mapped reaction parsing** with RDKit
- **Transition state graph construction** combining reactants and products
- **Bond change detection** (formed, broken, unchanged, order-changed)
- **Atom reactivity marking** based on chemical property changes
- **GNN encoder** using GINE (Graph Isomorphism Network with Edge features)
- **Contrastive loss functions** for CLIP-style training
- **Batch processing utilities** for efficient training

## Installation

### Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- RDKit

### Setup

1. Clone or navigate to the project directory:

```bash
cd project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

**Note:** Installing `torch-geometric` may require additional steps depending on your system. See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Project Structure

```
project/
├── data/                          # Data directory (empty by default)
├── reaction_encoder/
│   ├── __init__.py
│   ├── chem.py                    # Reaction parsing and RDKit utilities
│   ├── features.py                # Atom/bond featurization
│   ├── builder.py                 # Transition graph construction
│   ├── model.py                   # GNN model
│   ├── batch.py                   # Dataset and DataLoader
│   └── loss.py                    # Contrastive loss functions
├── scripts/
│   └── demo_encode.py             # Demo script
├── requirements.txt
└── README.md
```

## Quick Start

### Basic Usage

```python
import torch
from reaction_encoder.chem import parse_reaction_smiles
from reaction_encoder.builder import build_transition_graph
from reaction_encoder.model import ReactionGNN

# Reaction with atom mapping (required!)
rxn = "[N:1]=[N:2].[Ar:3]>>[N:1]-[N:2].[Ar:3]"

# Parse and build graph
reacts, prods = parse_reaction_smiles(rxn)
data = build_transition_graph(reacts, prods)

# Initialize model
model = ReactionGNN(
    x_dim=data.x.size(1),
    e_dim=data.edge_attr.size(1),
    hidden=128,
    layers=3,
    out_dim=256
)

# Encode reaction
model.eval()
with torch.no_grad():
    embedding = model(data)  # Shape: [1, 256]
```

### Run Demo

```bash
python scripts/demo_encode.py
```

This will:
1. Encode a single reaction and show the embedding
2. Compare multiple reactions using cosine similarity

## Key Components

### 1. Reaction Parsing (`chem.py`)

Parses reaction SMILES and creates atom mapping indices:

```python
from reaction_encoder.chem import parse_reaction_smiles, mapnums_index, bond_set

reacts, prods = parse_reaction_smiles("[C:1]=[O:2]>>[C:1][O:2]")
react_idx = mapnums_index(reacts)  # {mapnum: (mol_idx, atom_idx)}
react_bonds = bond_set(reacts)     # {(u,v): bond_properties}
```

### 2. Feature Extraction (`features.py`)

Extracts atomic features and detects changes:

```python
from reaction_encoder.features import atom_basic_features, atom_changed

features = atom_basic_features(atom)
# Returns: {Z, degree, formal_charge, is_aromatic, hyb, num_h, in_ring}

is_reactive = atom_changed(atom_react, atom_prod)
```

### 3. Graph Construction (`builder.py`)

Builds the transition state graph:

```python
from reaction_encoder.builder import build_transition_graph, diff_bonds

data = build_transition_graph(reacts, prods)
# Returns PyG Data with:
#   - x: node features [num_nodes, feature_dim]
#   - edge_index: connectivity [2, num_edges]
#   - edge_attr: edge features [num_edges, 6]
```

### 4. GNN Model (`model.py`)

Graph neural network encoder:

```python
from reaction_encoder.model import ReactionGNN

model = ReactionGNN(
    x_dim=16,        # Node feature dimension
    e_dim=6,         # Edge feature dimension
    hidden=128,      # Hidden layer size
    layers=3,        # Number of GNN layers
    out_dim=256      # Output embedding size
)

embedding = model(data)  # L2-normalized embedding
```

### 5. Batching (`batch.py`)

Dataset and DataLoader for multiple reactions:

```python
from reaction_encoder.batch import ReactionDataset, create_dataloader

reactions = ["[C:1]=[O:2]>>[C:1][O:2]", "[N:1]=[N:2]>>[N:1][N:2]"]
dataloader = create_dataloader(reactions, batch_size=32, shuffle=True)

for batch in dataloader:
    embeddings = model(batch)
```

### 6. Loss Functions (`loss.py`)

Contrastive learning losses:

```python
from reaction_encoder.loss import clip_loss

# For reaction-protein alignment
loss = clip_loss(protein_embeddings, reaction_embeddings, temperature=0.07)
```

## Graph Representation

### Node Features

Each node represents an atom (by map number) with features:
- **Reactant side**: [Z, degree, charge, aromatic, hyb, num_h, in_ring]
- **Product side**: [Z, degree, charge, aromatic, hyb, num_h, in_ring]
- **Existence flags**: [exists_in_reactant, exists_in_product]
- **Change flag**: [is_reactive]

Total: 16 features per node

### Edge Features

Each edge represents a bond with:
- Bond exists in reactant (1/0)
- Bond exists in product (1/0)
- Bond formed (1/0)
- Bond broken (1/0)
- Bond unchanged (1/0)
- Bond changed order (1/0)

Total: 6 features per edge

### Edge Changes

Edges are classified as:
- `UNCHANGED` (0): Bond exists in both with same properties
- `FORMED` (1): Bond only in products
- `BROKEN` (2): Bond only in reactants
- `CHANGED_ORDER` (3): Bond exists in both but different order/type

## Important Notes

### Atom Mapping Required

**All reactions must have atom mapping!** Use reaction SMILES with `:atomMapNumber` annotations:

✅ Good: `[C:1]=[O:2].[H:3][H:4]>>[C:1][O:2].[H:3][H:4]`

❌ Bad: `C=O.[H][H]>>CO.[H][H]`

If your reactions aren't mapped, use a tool like [RXNMapper](https://github.com/rxn4chemistry/rxnmapper) first.

### Edge Cases

- **Empty reactions** (no mapped atoms): Will create empty graphs
- **Reactions without changes**: All edges marked as UNCHANGED
- **Added/removed atoms**: Node features filled with zeros on missing side

## Future Enhancements

Potential improvements for V1:

- [ ] Automatic atom mapping integration (RXNMapper)
- [ ] Feature normalization/standardization
- [ ] Global attention pooling
- [ ] Pre-trained model weights
- [ ] Protein encoder integration for full CLIPZyme pipeline
- [ ] Training script with contrastive loss
- [ ] Support for reaction templates
- [ ] Visualization utilities

## References

- **CLIPZyme Paper**: [Reaction-Conditioned Virtual Screening of Enzymes](https://arxiv.org/abs/2402.06748)
- **CLIPZyme GitHub**: [pgmikhael/clipzyme](https://github.com/pgmikhael/clipzyme)
- **RDKit Documentation**: [rdkit.org](https://www.rdkit.org/)
- **PyTorch Geometric**: [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/)

## License

This is an educational implementation inspired by CLIPZyme. Please refer to the original CLIPZyme repository for their licensing terms.

## Citation

If you use this code, please cite the original CLIPZyme paper:

```bibtex
@article{mikhael2024clipzyme,
  title={CLIPZyme: Reaction-Conditioned Virtual Screening of Enzymes},
  author={Mikhael, Peter G. and others},
  journal={arXiv preprint arXiv:2402.06748},
  year={2024}
}
```
