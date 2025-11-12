# CLIPZyme - Refactored Architecture

## 🎉 What's New?

This project has been **completely refactored** with industry-standard design patterns and best practices. The result is cleaner, more maintainable, and easier to use code while preserving all original functionality.

## ✨ Key Improvements

| Aspect | Before | After |
|--------|---------|-------|
| **API Complexity** | Multiple encoders, manual setup | Single `CLIPZyme()` facade |
| **Configuration** | Hardcoded in 10+ files | YAML files + dataclasses |
| **Code Duplication** | `ProjectionHead` in 2 files | Unified in `common/modules.py` |
| **Demo Scripts** | 10+ overlapping scripts | 1 comprehensive demo |
| **Data Loading** | Inline in every script | Repository pattern |
| **Model Creation** | Manual instantiation | Factory + Builder patterns |
| **Architecture** | Tightly coupled | Loosely coupled via interfaces |

## 🚀 Quick Start (3 Lines!)

```python
from clipzyme import CLIPZyme

clipzyme = CLIPZyme()
similarity = clipzyme.compute_similarity(["MSKGEEL..."], ["[N:1]=[N:2]>>[N:1][N:2]"])
print(similarity)
```

That's it! The Facade pattern makes it super simple.

## 📁 New Project Structure

```
project/
├── common/                      # 🆕 Shared components
│   ├── constants.py            # All magic numbers centralized
│   ├── interfaces.py           # Abstract base classes (Strategy Pattern)
│   ├── modules.py              # ProjectionHead, MLP (no duplication!)
│   ├── factory.py              # Factory Pattern for dynamic creation
│   └── __init__.py
│
├── config/                      # 🆕 Configuration system
│   ├── config.py               # Type-safe dataclasses
│   └── __init__.py
│
├── configs/                     # 🆕 YAML configurations
│   ├── default.yaml            # Default settings
│   └── clipzyme_faithful.yaml  # Paper reproduction
│
├── data/                        # 🆕 Data access layer
│   ├── repositories.py         # Repository Pattern
│   ├── proteins.csv            # ✅ Data in CSV (not inline!)
│   ├── reactions_extended.csv
│   ├── enzyme_reactions.csv
│   └── __init__.py
│
├── models/                      # 🆕 Model layer
│   ├── clipzyme.py             # Unified CLIPZyme model
│   ├── builder.py              # Builder Pattern
│   └── __init__.py
│
├── clipzyme/                    # 🆕 High-level API
│   ├── facade.py               # Facade Pattern (simplified API)
│   └── __init__.py
│
├── protein_encoder/             # ✅ Existing (unchanged)
├── reaction_encoder/            # ✅ Existing (unchanged)
│
├── demo.py                      # 🆕 Unified demo (replaces 10+ scripts)
├── REFACTORING_GUIDE.md         # 🆕 Complete refactoring guide
└── USAGE_EXAMPLES.md            # 🆕 20+ usage examples
```

## 🎨 Design Patterns Implemented

### 1. **Facade Pattern** - Simple API

```python
from clipzyme import CLIPZyme

clipzyme = CLIPZyme()
similarity = clipzyme.compute_similarity(proteins, reactions)
best_matches = clipzyme.find_best_reactions_for_protein(protein, reactions, top_k=5)
```

**Why?** Hides complexity, perfect for quick prototyping.

### 2. **Strategy Pattern** - Interchangeable Encoders

```python
class ProteinEncoder(ABC):
    @abstractmethod
    def encode(self, sequences) -> Tensor: pass

class ReactionEncoder(ABC):
    @abstractmethod
    def encode(self, reactions) -> Tensor: pass
```

**Why?** Swap encoder implementations without changing code.

### 3. **Factory Pattern** - Dynamic Creation

```python
from common.factory import create_protein_encoder

encoder = create_protein_encoder(config)  # Creates ESM2 or EGNN based on config
```

**Why?** Configuration-driven instantiation, no hardcoded values.

### 4. **Builder Pattern** - Fluent Construction

```python
model = (CLIPZymeBuilder()
         .with_protein_encoder_config(protein_config)
         .with_reaction_encoder_config(reaction_config)
         .with_temperature(0.07)
         .on_device('cuda')
         .build())
```

**Why?** Readable, flexible model construction.

### 5. **Repository Pattern** - Clean Data Access

```python
from data.repositories import ProteinRepository

repo = ProteinRepository('data/proteins.csv')
proteins = repo.load_all(max_length=100, name_contains='GFP')
```

**Why?** Decoupled data access with caching and filtering.

### 6. **Configuration System** - YAML + Dataclasses

```yaml
# configs/my_experiment.yaml
protein_encoder:
  type: ESM2
  proj_dim: 512

reaction_encoder:
  type: Enhanced
  hidden_dim: 256

training:
  learning_rate: 0.0001
  batch_size: 32
```

```python
config = load_config('configs/my_experiment.yaml')
model = create_model(config)
```

**Why?** No hardcoded values, reproducible research.

## 📖 Usage Examples

### Example 1: Simplest Usage (Facade)

```python
from clipzyme import CLIPZyme

# Initialize
clipzyme = CLIPZyme()

# Load data from CSV
proteins = clipzyme.load_proteins_from_csv(max_length=100)
reactions = clipzyme.load_reactions_from_csv()

# Encode and compare
similarity = clipzyme.compute_similarity(
    [p.sequence for p in proteins],
    [r.reaction_smiles for r in reactions]
)

# Find matches
matches = clipzyme.find_best_reactions_for_protein(
    proteins[0].sequence,
    [r.reaction_smiles for r in reactions],
    top_k=5
)

for match in matches:
    print(f"Rank {match['rank']}: {match['score']:.4f}")
```

### Example 2: Configuration-Driven

```python
from clipzyme import CLIPZyme

# Use different configurations
clipzyme_small = CLIPZyme(config='default')          # Fast, 35M params
clipzyme_paper = CLIPZyme(config='faithful')         # Paper, 650M params
clipzyme_custom = CLIPZyme(config='configs/my.yaml') # Custom

# Or modify configuration
from config.config import CLIPZymeConfig

config = CLIPZymeConfig.default()
config.protein_encoder.proj_dim = 512
config.training.learning_rate = 1e-5

# Save for reproducibility
config.to_yaml('configs/experiment1.yaml')
```

### Example 3: Advanced (Builder)

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
             proj_dim=256
         ))
         .with_temperature(0.07)
         .on_device('cuda')
         .build())
```

### Example 4: Data Loading (Repository)

```python
from data.repositories import ProteinRepository, ReactionRepository

# Proteins
protein_repo = ProteinRepository('data/proteins.csv')

short_proteins = protein_repo.load_all(max_length=100)
gfp_proteins = protein_repo.load_all(name_contains='GFP')
random_proteins = protein_repo.get_random_sample(n=5)

# Reactions
reaction_repo = ReactionRepository('data/reactions_extended.csv')

reductions = reaction_repo.load_all(name_contains='reduction')
specific = reaction_repo.load_by_id('N=N reduction')
```

## 🎯 Migration Guide

### Old Code (Before Refactoring)

```python
# Scattered across multiple files, hardcoded values
from protein_encoder.esm_model import ProteinEncoderESM2
from reaction_encoder.model_enhanced import ReactionGNNEnhanced

protein_encoder = ProteinEncoderESM2(
    plm_name="facebook/esm2_t12_35M_UR50D",  # hardcoded
    pooling="attention",
    proj_dim=256,
)

reaction_encoder = ReactionGNNEnhanced(
    x_dim=7,  # magic number
    e_dim=3,  # magic number
    hidden=128,
)

# Manual CSV loading
import csv
proteins = []
with open('data/proteins.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        proteins.append(row['sequence'])
```

### New Code (After Refactoring)

```python
# One-line initialization, no hardcoded values!
from clipzyme import CLIPZyme

clipzyme = CLIPZyme(config='default')

# Clean data loading
proteins = clipzyme.load_proteins_from_csv(max_length=100)
reactions = clipzyme.load_reactions_from_csv()

# Simple operations
similarity = clipzyme.compute_similarity(
    [p.sequence for p in proteins],
    [r.reaction_smiles for r in reactions]
)
```

## 🎬 Running the Demo

```bash
# Run all demos
python demo.py

# Run specific demo
python demo.py --demo simple       # Facade Pattern
python demo.py --demo config       # Configuration System
python demo.py --demo builder      # Builder Pattern
python demo.py --demo repository   # Repository Pattern
python demo.py --demo factory      # Factory Pattern
python demo.py --demo workflow     # Complete workflow
```

## 📚 Documentation

- **`REFACTORING_GUIDE.md`** - Complete guide to the refactoring
- **`USAGE_EXAMPLES.md`** - 20+ practical examples
- **`configs/`** - YAML configuration examples
- **`demo.py`** - Interactive demos

## 🎁 Benefits Summary

### For Users
- ✅ **Simpler API**: `CLIPZyme()` vs manual encoder setup
- ✅ **Better docs**: Comprehensive examples and guides
- ✅ **Easier configuration**: YAML files vs code editing

### For Developers
- ✅ **No duplication**: Shared modules in `common/`
- ✅ **Clean architecture**: Interfaces, factories, repositories
- ✅ **Maintainable**: Single responsibility, separation of concerns
- ✅ **Testable**: Easy to mock and test
- ✅ **Extensible**: Add new encoders without changing existing code

### For Researchers
- ✅ **Reproducible**: Configuration files for experiments
- ✅ **Flexible**: Easy to try different architectures
- ✅ **Documented**: Clear examples and guides

## 🔧 What's Next?

The refactoring is **complete** for the core system. You can now:

1. **Use the Facade API** for quick experiments
2. **Create custom configurations** in YAML
3. **Add your own encoders** using the interfaces
4. **Load custom data** using repositories
5. **Build complex models** with the Builder

All while enjoying clean, maintainable, professional code! 🎉

## 📝 File Mapping

Old scattered code → New organized code:

| Old | New |
|-----|-----|
| `scripts/demo_*.py` (10 files) | `demo.py` (1 file) |
| Inline data in scripts | `data/*.csv` + `data/repositories.py` |
| Hardcoded values everywhere | `common/constants.py` + `configs/*.yaml` |
| Duplicated `ProjectionHead` | `common/modules.py` |
| Manual model creation | `common/factory.py` + `models/builder.py` |
| No high-level API | `clipzyme/facade.py` |

## 🏆 Summary

**Before**: Research prototype with duplicated code and hardcoded values
**After**: Production-ready system with design patterns and clean architecture

**All original functionality preserved!** ✅

Enjoy your beautifully refactored CLIPZyme! 🚀
