# CLIPZyme Usage Examples

Complete guide with practical examples for using the refactored CLIPZyme system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Facade API (Simplest)](#facade-api-simplest)
3. [Configuration System](#configuration-system)
4. [Data Loading (Repository Pattern)](#data-loading-repository-pattern)
5. [Custom Model Building (Builder Pattern)](#custom-model-building-builder-pattern)
6. [Factory Pattern](#factory-pattern)
7. [Complete Workflows](#complete-workflows)

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install torch torch-geometric transformers rdkit-pypi pyyaml

# Navigate to project
cd project/
```

### Simplest Example (3 lines!)

```python
from clipzyme import CLIPZyme

clipzyme = CLIPZyme()
similarity = clipzyme.compute_similarity(["MSKGEEL..."], ["[N:1]=[N:2]>>[N:1][N:2]"])
print(similarity)
```

---

## Facade API (Simplest)

The `CLIPZyme` facade provides the easiest way to use the system.

### Example 1: Basic Encoding

```python
from clipzyme import CLIPZyme

# Initialize (auto-detects GPU/CPU)
clipzyme = CLIPZyme()

# Encode proteins
proteins = [
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEG",
    "MAHHHHHHSLENPLKQFGPVVVNQQWKK"
]
protein_embeddings = clipzyme.encode_proteins(proteins)
print(f"Protein embeddings: {protein_embeddings.shape}")  # (2, 256)

# Encode reactions
reactions = [
    "[N:1]=[N:2]>>[N:1][N:2]",
    "[C:1]=[C:2]>>[C:1][C:2]"
]
reaction_embeddings = clipzyme.encode_reactions(reactions)
print(f"Reaction embeddings: {reaction_embeddings.shape}")  # (2, 256)
```

### Example 2: Compute Similarity

```python
from clipzyme import CLIPZyme

clipzyme = CLIPZyme()

proteins = ["MSKGEEL...", "MAHHHHH..."]
reactions = ["[N:1]=[N:2]>>[N:1][N:2]", "[C:1]=[C:2]>>[C:1][C:2]"]

# Compute all pairwise similarities
similarity_matrix = clipzyme.compute_similarity(proteins, reactions)
print(similarity_matrix)
# [[0.523, 0.612],
#  [0.701, 0.489]]
```

### Example 3: Find Best Matches

```python
from clipzyme import CLIPZyme

clipzyme = CLIPZyme()

# Find best reactions for a protein
protein = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEG"
reactions = [
    "[N:1]=[N:2]>>[N:1][N:2]",
    "[C:1]=[C:2]>>[C:1][C:2]",
    "[C:1]#[C:2]>>[C:1]=[C:2]",
]

matches = clipzyme.find_best_reactions_for_protein(protein, reactions, top_k=3)

for match in matches:
    print(f"Rank {match['rank']}: {match['reaction']}")
    print(f"  Score: {match['score']:.4f}\n")
```

### Example 4: Load Data from CSV

```python
from clipzyme import CLIPZyme

clipzyme = CLIPZyme()

# Load proteins with filtering
proteins = clipzyme.load_proteins_from_csv(
    max_length=100,  # Only short proteins
    name_contains='GFP'  # Only GFP proteins
)

print(f"Loaded {len(proteins)} proteins")
for protein in proteins:
    print(f"  - {protein.name}: {protein.length} aa")

# Load reactions
reactions = clipzyme.load_reactions_from_csv(
    name_contains='reduction'
)

print(f"\nLoaded {len(reactions)} reactions")

# Load enzyme-reaction pairs
pairs = clipzyme.load_enzyme_reactions_from_csv(max_length=500)
print(f"\nLoaded {len(pairs)} enzyme-reaction pairs")
```

---

## Configuration System

Use YAML files for reproducible experiments.

### Example 5: Use Different Configurations

```python
from clipzyme import CLIPZyme

# Option 1: Default configuration
clipzyme_default = CLIPZyme(config='default')

# Option 2: CLIPZyme faithful (paper reproduction)
# Warning: Downloads 650M parameter model!
clipzyme_faithful = CLIPZyme(config='faithful')

# Option 3: Custom YAML file
clipzyme_custom = CLIPZyme(config='configs/my_experiment.yaml')

# Option 4: Specify device
clipzyme_gpu = CLIPZyme(config='default', device='cuda')
```

### Example 6: Create Custom Configuration

```python
from config.config import CLIPZymeConfig

# Start from default
config = CLIPZymeConfig.default()

# Customize settings
config.protein_encoder.proj_dim = 512
config.reaction_encoder.hidden_dim = 256
config.training.learning_rate = 1e-5
config.training.batch_size = 32

# Save for later use
config.to_yaml('configs/my_experiment.yaml')

# Use it
from clipzyme import CLIPZyme
clipzyme = CLIPZyme(config='configs/my_experiment.yaml')
```

### Example 7: Modify YAML Directly

Create `configs/experiment.yaml`:

```yaml
experiment_name: my_experiment
description: Testing with larger projection dimension
random_seed: 42

protein_encoder:
  type: ESM2
  plm_name: facebook/esm2_t12_35M_UR50D
  pooling: attention
  proj_dim: 512  # Increased from 256
  dropout: 0.1

reaction_encoder:
  type: Enhanced
  hidden_dim: 256  # Increased from 128
  num_layers: 4  # Increased from 3
  proj_dim: 512  # Match protein encoder
  use_attention_pool: true

training:
  batch_size: 32
  learning_rate: 0.00005  # Decreased
  temperature: 0.05  # Decreased
```

Then use it:

```python
from clipzyme import CLIPZyme

clipzyme = CLIPZyme(config='configs/experiment.yaml')
```

---

## Data Loading (Repository Pattern)

Clean, filtered data access without SQL.

### Example 8: Protein Repository

```python
from data.repositories import ProteinRepository

repo = ProteinRepository('data/proteins.csv')

# Count proteins
print(f"Total proteins: {repo.count()}")
print(f"Short proteins: {repo.count(max_length=50)}")
print(f"Long proteins: {repo.count(min_length=200)}")
print(f"GFP proteins: {repo.count(name_contains='GFP')}")

# Load with filters
short_proteins = repo.load_all(max_length=100)
bacterial_proteins = repo.load_all(description_contains='coli')

# Load specific protein
gfp = repo.load_by_id('GFP-full')
if gfp:
    print(f"{gfp.name}: {gfp.length} aa")
    print(f"Sequence: {gfp.sequence}")

# Random sampling
random_sample = repo.get_random_sample(n=5, max_length=100)
```

### Example 9: Reaction Repository

```python
from data.repositories import ReactionRepository

repo = ReactionRepository('data/reactions_extended.csv')

# Filter by name
reductions = repo.load_all(name_contains='reduction')
hydrogenations = repo.load_all(name_contains='hydrogenation')

print(f"Found {len(reductions)} reduction reactions")

# Get specific reaction
reaction = repo.load_by_id('N=N reduction')
if reaction:
    print(f"SMILES: {reaction.reaction_smiles}")
    print(f"Description: {reaction.description}")

# Random sample
random_reactions = repo.get_random_sample(n=10)
```

### Example 10: Enzyme-Reaction Pairs

```python
from data.repositories import EnzymeReactionRepository

repo = EnzymeReactionRepository('data/enzyme_reactions.csv')

# Load pairs with filters
pairs = repo.load_all(
    max_length=300,
    name_contains='reductase'
)

print(f"Found {len(pairs)} reductase pairs")

for pair in pairs:
    print(f"\nEnzyme: {pair.name}")
    print(f"Sequence length: {len(pair.sequence)} aa")
    print(f"Reaction: {pair.reaction}")
    print(f"Description: {pair.description}")

# Get all sequences for encoding
all_proteins = repo.get_proteins()
all_reactions = repo.get_reactions()
```

---

## Custom Model Building (Builder Pattern)

Fluent API for advanced model construction.

### Example 11: Basic Builder Usage

```python
from models.builder import CLIPZymeBuilder
from config.config import ProteinEncoderConfig, ReactionEncoderConfig

# Build model step-by-step
model = (CLIPZymeBuilder()
         .with_protein_encoder_config(ProteinEncoderConfig(
             type='ESM2',
             plm_name='facebook/esm2_t12_35M_UR50D',
             pooling='attention',
             proj_dim=256
         ))
         .with_reaction_encoder_config(ReactionEncoderConfig(
             type='Enhanced',
             hidden_dim=128,
             num_layers=3,
             proj_dim=256
         ))
         .with_temperature(0.07)
         .on_device('cuda')
         .build())

print(f"Built model on {model.device}")
```

### Example 12: Builder from Preset

```python
from models.builder import CLIPZymeBuilder

# Start from preset and modify
model = (CLIPZymeBuilder()
         .from_preset('default')
         .with_temperature(0.05)  # Override temperature
         .on_device('cuda')
         .build())
```

### Example 13: Builder from YAML

```python
from models.builder import CLIPZymeBuilder

# Load from YAML and modify
model = (CLIPZymeBuilder()
         .from_yaml('configs/default.yaml')
         .with_learnable_temperature(True)  # Make temperature learnable
         .on_device('cuda')
         .build())

print(f"Temperature is learnable: {model._learnable_temperature}")
```

### Example 14: Convenience Function

```python
from models.builder import build_clipzyme_model

# Quick building
model1 = build_clipzyme_model()  # Default
model2 = build_clipzyme_model('faithful', device='cuda')  # Preset
model3 = build_clipzyme_model('configs/my_config.yaml')  # YAML
```

---

## Factory Pattern

Dynamic encoder creation from configuration.

### Example 15: Create Encoders

```python
from common.factory import create_protein_encoder, create_reaction_encoder
from config.config import ProteinEncoderConfig, ReactionEncoderConfig

# Create protein encoder
protein_config = ProteinEncoderConfig(
    type='ESM2',
    plm_name='facebook/esm2_t12_35M_UR50D',
    proj_dim=256
)
protein_encoder = create_protein_encoder(protein_config)

# Create reaction encoder
reaction_config = ReactionEncoderConfig(
    type='Enhanced',
    hidden_dim=128,
    proj_dim=256
)
reaction_encoder = create_reaction_encoder(reaction_config)

# Use encoders
sequences = ["MSKGEEL...", "MAHHHHH..."]
embeddings = protein_encoder.encode(sequences, device='cpu')
print(f"Embeddings: {embeddings.shape}")
```

### Example 16: Switch Encoder Types

```python
from common.factory import create_reaction_encoder
from config.config import ReactionEncoderConfig

# Try different encoder types
for encoder_type in ['GNN', 'Enhanced', 'DMPNN']:
    config = ReactionEncoderConfig(
        type=encoder_type,
        hidden_dim=128,
        proj_dim=256
    )

    encoder = create_reaction_encoder(config)
    print(f"Created {encoder_type}: {type(encoder).__name__}")
```

---

## Complete Workflows

End-to-end examples.

### Example 17: Protein-Reaction Screening

```python
from clipzyme import CLIPZyme

# Initialize system
clipzyme = CLIPZyme(config='default', device='cuda')

# Load candidate proteins and reactions
proteins = clipzyme.load_proteins_from_csv(max_length=500)
reactions = clipzyme.load_reactions_from_csv()

# Screen all combinations
protein_seqs = [p.sequence for p in proteins]
reaction_smiles = [r.reaction_smiles for r in reactions]

similarity = clipzyme.compute_similarity(protein_seqs, reaction_smiles)

# Find best pairs
best_pairs = []
for i, protein in enumerate(proteins):
    best_reaction_idx = similarity[i].argmax()
    best_score = similarity[i, best_reaction_idx]

    best_pairs.append({
        'protein': protein.name,
        'reaction': reactions[best_reaction_idx].reaction_name,
        'score': best_score
    })

# Sort by score
best_pairs.sort(key=lambda x: x['score'], reverse=True)

print("Top 10 Protein-Reaction Pairs:")
for i, pair in enumerate(best_pairs[:10], 1):
    print(f"{i}. {pair['protein']} <-> {pair['reaction']}")
    print(f"   Score: {pair['score']:.4f}\n")
```

### Example 18: Enzyme Discovery Pipeline

```python
from clipzyme import CLIPZyme
import numpy as np

clipzyme = CLIPZyme()

# Target reaction
target_reaction = "[C:1]=[C:2]>>[C:1]([H])[C:2]([H])"

# Candidate enzymes from database
enzymes = clipzyme.load_proteins_from_csv(
    name_contains='reductase',
    max_length=600
)

print(f"Screening {len(enzymes)} candidate enzymes...")

# Compute similarities
enzyme_seqs = [e.sequence for e in enzymes]
similarities = clipzyme.compute_similarity(enzyme_seqs, [target_reaction])

# Rank enzymes
results = []
for i, enzyme in enumerate(enzymes):
    results.append({
        'name': enzyme.name,
        'score': similarities[i, 0],
        'length': enzyme.length
    })

results.sort(key=lambda x: x['score'], reverse=True)

print("\nTop candidate enzymes:")
for i, result in enumerate(results[:5], 1):
    print(f"{i}. {result['name']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Length: {result['length']} aa\n")
```

### Example 19: Batch Processing

```python
from clipzyme import CLIPZyme

clipzyme = CLIPZyme()

# Load large dataset
pairs = clipzyme.load_enzyme_reactions_from_csv()
print(f"Processing {len(pairs)} pairs...")

# Extract sequences
proteins = [p.sequence for p in pairs]
reactions = [p.reaction for p in pairs]

# Encode in batches
protein_embeddings = clipzyme.encode_proteins(proteins, batch_size=32)
reaction_embeddings = clipzyme.encode_reactions(reactions, batch_size=32)

print(f"Protein embeddings: {protein_embeddings.shape}")
print(f"Reaction embeddings: {reaction_embeddings.shape}")

# Save embeddings
import numpy as np
np.save('protein_embeddings.npy', protein_embeddings)
np.save('reaction_embeddings.npy', reaction_embeddings)

print("Embeddings saved!")
```

### Example 20: Model Evaluation

```python
from clipzyme import CLIPZyme
import numpy as np

clipzyme = CLIPZyme()

# Load test set
test_pairs = clipzyme.load_enzyme_reactions_from_csv()[:100]

# Encode
proteins = [p.sequence for p in test_pairs]
reactions = [p.reaction for p in test_pairs]

similarity = clipzyme.compute_similarity(proteins, reactions)

# Compute retrieval accuracy (Recall@K)
def recall_at_k(similarity_matrix, k=1):
    n = similarity_matrix.shape[0]
    correct = 0

    for i in range(n):
        # Top K indices
        top_k = np.argsort(similarity_matrix[i])[::-1][:k]
        if i in top_k:  # Diagonal should be in top K
            correct += 1

    return correct / n

r1 = recall_at_k(similarity, k=1)
r5 = recall_at_k(similarity, k=5)
r10 = recall_at_k(similarity, k=10)

print(f"Recall@1:  {r1:.2%}")
print(f"Recall@5:  {r5:.2%}")
print(f"Recall@10: {r10:.2%}")
```

---

## Summary

This refactored CLIPZyme provides multiple levels of abstraction:

1. **Facade** (`CLIPZyme`): Simplest, for quick prototyping
2. **Builder**: Flexible model construction
3. **Factory**: Dynamic encoder creation
4. **Repository**: Clean data access
5. **Configuration**: YAML-based settings

Choose the level that fits your needs!

For more examples, run:
```bash
python demo.py --demo all
```
