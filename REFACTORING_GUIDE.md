# CLIPZyme Refactoring Guide

## Overview

This project has been completely refactored to implement industry-standard design patterns and best practices. The refactoring improves maintainability, testability, extensibility, and ease of use.

## What Changed?

### Before
- ❌ Duplicated code (ProjectionHead in 2 files)
- ❌ Hardcoded values scattered everywhere
- ❌ 10+ demo scripts with overlapping functionality
- ❌ Tight coupling between components
- ❌ God classes (200+ line trainers)
- ❌ No configuration system
- ❌ Inline data in scripts

### After
- ✅ Unified shared modules (common/)
- ✅ Centralized constants and configuration
- ✅ Single comprehensive demo
- ✅ Loose coupling via interfaces
- ✅ Single responsibility classes
- ✅ YAML-based configuration
- ✅ All data in CSV files

## New Architecture

```
project/
├── common/                 # Shared components
│   ├── constants.py       # All hardcoded values
│   ├── interfaces.py      # Abstract base classes
│   ├── modules.py         # ProjectionHead, MLP
│   ├── factory.py         # Factory pattern
│   └── __init__.py
│
├── config/                 # Configuration system
│   ├── config.py          # Dataclass configurations
│   └── __init__.py
│
├── configs/                # YAML configurations
│   ├── default.yaml       # Default settings
│   └── clipzyme_faithful.yaml  # Paper reproduction
│
├── data/                   # Data access layer
│   ├── repositories.py    # Repository pattern
│   ├── *.csv              # All data in CSV format
│   └── __init__.py
│
├── models/                 # Model layer
│   ├── clipzyme.py        # Unified CLIPZyme model
│   ├── builder.py         # Builder pattern
│   └── __init__.py
│
├── clipzyme/               # High-level API
│   ├── facade.py          # Facade pattern
│   └── __init__.py
│
├── protein_encoder/        # (existing)
├── reaction_encoder/       # (existing)
├── trainer/                # Training components
└── demo.py                 # Unified demo script
```

## Design Patterns Implemented

### 1. **Facade Pattern** (`clipzyme/facade.py`)

Simplifies the API for common use cases.

```python
from clipzyme import CLIPZyme

# One-line initialization
clipzyme = CLIPZyme()

# Simple operations
proteins = ["MSKGEEL...", "MAHHHHH..."]
reactions = ["[N:1]=[N:2]>>[N:1][N:2]"]

similarity = clipzyme.compute_similarity(proteins, reactions)
```

**Benefits:**
- Simple API for users
- Hides complexity
- Quick prototyping

### 2. **Strategy Pattern** (`common/interfaces.py`)

Defines common interfaces for encoders.

```python
class ProteinEncoder(ABC):
    @abstractmethod
    def encode(self, sequences, **kwargs) -> torch.Tensor:
        pass

class ReactionEncoder(ABC):
    @abstractmethod
    def encode(self, reactions, **kwargs) -> torch.Tensor:
        pass
```

**Benefits:**
- Swap encoder implementations easily
- Consistent API
- Better testing

### 3. **Factory Pattern** (`common/factory.py`)

Creates encoders dynamically from configuration.

```python
from common.factory import create_protein_encoder
from config.config import ProteinEncoderConfig

config = ProteinEncoderConfig(type='ESM2', proj_dim=256)
encoder = create_protein_encoder(config)
```

**Benefits:**
- No hardcoded instantiation
- Configuration-driven
- Easy to extend

### 4. **Builder Pattern** (`models/builder.py`)

Fluent API for model construction.

```python
from models.builder import CLIPZymeBuilder

model = (CLIPZymeBuilder()
         .with_protein_encoder_config(protein_config)
         .with_reaction_encoder_config(reaction_config)
         .with_temperature(0.07)
         .on_device('cuda')
         .build())
```

**Benefits:**
- Readable code
- Step-by-step construction
- Flexible configuration

### 5. **Repository Pattern** (`data/repositories.py`)

Clean data access abstraction.

```python
from data.repositories import ProteinRepository

repo = ProteinRepository('data/proteins.csv')

# Filtering
short_proteins = repo.load_all(max_length=100)
gfp_proteins = repo.load_all(name_contains='GFP')

# Sampling
random_proteins = repo.get_random_sample(n=5)
```

**Benefits:**
- Decoupled data access
- Cachingbuilt-in
- Easy filtering/querying

### 6. **Configuration System** (`config/`)

YAML-based configuration with type safety.

```python
from config.config import CLIPZymeConfig, load_config

# From YAML
config = load_config('configs/default.yaml')

# From preset
config = CLIPZymeConfig.default()
config = CLIPZymeConfig.clipzyme_faithful()

# Modify and save
config.training.learning_rate = 1e-5
config.to_yaml('configs/my_config.yaml')
```

**Benefits:**
- No hardcoded values
- Reproducible research
- Easy experimentation

## Migration Guide

### Old Way (Before Refactoring)

```python
# Scattered across multiple files
from protein_encoder.esm_model import ProteinEncoderESM2
from reaction_encoder.model_enhanced import ReactionGNNEnhanced

# Hardcoded values
protein_encoder = ProteinEncoderESM2(
    plm_name="facebook/esm2_t12_35M_UR50D",  # hardcoded
    pooling="attention",
    proj_dim=256,
    dropout=0.1
)

reaction_encoder = ReactionGNNEnhanced(
    x_dim=7,  # magic number
    e_dim=3,  # magic number
    hidden=128,
    layers=3,
    out_dim=256
)

# Manual data loading in every script
import csv
proteins = []
with open('data/proteins.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        proteins.append(row['sequence'])
```

### New Way (After Refactoring)

```python
# Simple Facade API
from clipzyme import CLIPZyme

clipzyme = CLIPZyme(config='default')  # or 'faithful' or path to YAML

# Load data with Repository
proteins = clipzyme.load_proteins_from_csv(max_length=100)
reactions = clipzyme.load_reactions_from_csv()

# Encode and compute similarity
similarity = clipzyme.compute_similarity(
    [p.sequence for p in proteins],
    [r.reaction_smiles for r in reactions]
)
```

**Or, for advanced usage:**

```python
# Configuration-driven approach
from config.config import load_config
from common.factory import create_model

config = load_config('configs/my_config.yaml')
model = create_model(config)

# Or use Builder for fine-grained control
from models.builder import CLIPZymeBuilder

model = (CLIPZymeBuilder()
         .from_yaml('configs/default.yaml')
         .with_temperature(0.05)  # override
         .on_device('cuda')
         .build())
```

## Quick Start

### 1. Simple Usage (Facade)

```python
from clipzyme import CLIPZyme

# Initialize with defaults
clipzyme = CLIPZyme()

# Or with specific config
clipzyme = CLIPZyme(config='configs/default.yaml')

# Encode and compare
proteins = ["MSKGEEL..."]
reactions = ["[N:1]=[N:2]>>[N:1][N:2]"]

similarity = clipzyme.compute_similarity(proteins, reactions)
print(f"Similarity: {similarity[0,0]:.4f}")

# Find best matches
matches = clipzyme.find_best_reactions_for_protein(
    proteins[0],
    reactions,
    top_k=5
)

for match in matches:
    print(f"Rank {match['rank']}: {match['score']:.4f}")
```

### 2. Configuration-Based

```python
from config.config import CLIPZymeConfig
from models.builder import build_clipzyme_model

# Create or load config
config = CLIPZymeConfig.default()

# Customize
config.protein_encoder.proj_dim = 512
config.training.learning_rate = 1e-5

# Build model
model = build_clipzyme_model(config, device='cuda')

# Save config for reproducibility
config.to_yaml('configs/experiment1.yaml')
```

### 3. Data Loading (Repository)

```python
from data.repositories import (
    ProteinRepository,
    ReactionRepository,
    EnzymeReactionRepository
)

# Load proteins with filtering
protein_repo = ProteinRepository('data/proteins.csv')
short_proteins = protein_repo.load_all(max_length=100)
gfp_proteins = protein_repo.load_all(name_contains='GFP')

# Load reactions
reaction_repo = ReactionRepository('data/reactions_extended.csv')
reductions = reaction_repo.load_all(name_contains='reduction')

# Load enzyme-reaction pairs
pair_repo = EnzymeReactionRepository('data/enzyme_reactions.csv')
pairs = pair_repo.load_all(max_length=500)
```

### 4. Custom Model Building (Builder)

```python
from models.builder import CLIPZymeBuilder
from config.config import ProteinEncoderConfig, ReactionEncoderConfig

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
         .with_learnable_temperature(False)
         .on_device('cuda')
         .build())
```

## Running the Demo

```bash
# Run all demos
python demo.py

# Run specific demo
python demo.py --demo simple
python demo.py --demo config
python demo.py --demo builder
python demo.py --demo repository
python demo.py --demo factory
python demo.py --demo workflow
```

## Benefits of Refactoring

### Code Quality
- ✅ **DRY**: No duplicated code
- ✅ **SOLID**: Single responsibility, open/closed, dependency inversion
- ✅ **Clean**: Clear separation of concerns
- ✅ **Testable**: Easy to mock and test

### Maintainability
- ✅ **Centralized**: Constants, configs, shared modules in one place
- ✅ **Documented**: Clear interfaces and docstrings
- ✅ **Consistent**: Uniform coding patterns

### Extensibility
- ✅ **Easy to add**: New encoders, pooling methods, loss functions
- ✅ **Pluggable**: Swap implementations without changing code
- ✅ **Configurable**: Everything driven by YAML configs

### Usability
- ✅ **Simple API**: Facade pattern for common use cases
- ✅ **Flexible**: Builder pattern for advanced usage
- ✅ **Well-documented**: Examples and guides

## Next Steps

1. **Experiment with configurations**
   ```bash
   # Edit configs/default.yaml
   # Change hyperparameters, model types, etc.
   ```

2. **Add your own data**
   ```bash
   # Add rows to data/*.csv files
   # Or create new CSV files
   ```

3. **Create custom models**
   ```python
   # Implement ProteinEncoder or ReactionEncoder
   # Register in Factory
   ```

4. **Train models**
   ```python
   # Use the new configuration system
   # Training code to be refactored next
   ```

## Summary

This refactoring transforms CLIPZyme from a research prototype into a production-ready codebase:

- **Before**: 10+ scripts, duplicated code, hardcoded values
- **After**: Clean architecture, design patterns, configuration system

The new architecture is:
- **Easier to use** (Facade API)
- **Easier to maintain** (No duplication)
- **Easier to extend** (Interfaces + Factory)
- **Easier to configure** (YAML configs)
- **Easier to test** (Dependency injection)

All while preserving the original scientific functionality!
