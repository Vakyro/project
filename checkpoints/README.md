# CLIPZyme Checkpoint Management

Complete system for downloading, loading, and managing CLIPZyme model checkpoints.

## 📋 Overview

The checkpoint system provides:
- **Download**: Automatic download from Zenodo
- **Load**: Universal loader for official and local checkpoints
- **Convert**: Convert between checkpoint formats
- **Validate**: Verify checkpoint integrity
- **Inspect**: Examine checkpoint contents
- **Compare**: Compare different checkpoints

## 🚀 Quick Start

### Load Official Pretrained Model

```python
from models import load_pretrained

# Automatically downloads from Zenodo if not cached
model = load_pretrained("clipzyme", device="cuda")

# Use immediately
embeddings = model.encode_reactions(["[C:1]=[O:2]>>[C:1]-[O:2]"])
```

### Load Checkpoint File

```python
from models import load_checkpoint

# Load from any checkpoint format
model = load_checkpoint("data/checkpoints/clipzyme_model.ckpt", device="cuda")
model.eval()
```

## 📦 Command Line Interface

### Download Official Checkpoint

```bash
# Download model checkpoint (2.4 GB)
python scripts/manage_checkpoints.py download --output data/checkpoints

# Download all CLIPZyme files (model + data + splits)
python scripts/manage_checkpoints.py download --all

# Download only data
python scripts/manage_checkpoints.py download --file data
```

### Inspect Checkpoint

```bash
python scripts/manage_checkpoints.py inspect --checkpoint data/checkpoints/clipzyme_model.ckpt
```

Output:
```
CHECKPOINT INFORMATION
==============================================================
File: data/checkpoints/clipzyme_model.ckpt
Size: 2451.23 MB
Format: pytorch_lightning
Total Parameters: 654,987,321
Epoch: 29
Global Step: 45,600

Largest Parameters:
  - protein_encoder.esm_model.layers.33.self_attn.weight: 2,097,152
  - reaction_encoder.dmpnn.layer_0.weight: 1,048,576
  ...
```

### Validate Checkpoint

```bash
# Validate checkpoint integrity
python scripts/manage_checkpoints.py validate \
    --checkpoint data/checkpoints/clipzyme_model.ckpt \
    --test-forward

# Check expected parameter count
python scripts/manage_checkpoints.py validate \
    --checkpoint data/checkpoints/clipzyme_model.ckpt \
    --expected-params 654987321
```

###Convert Checkpoint

```bash
# Convert official to local format
python scripts/manage_checkpoints.py convert \
    --input data/checkpoints/official.ckpt \
    --output data/checkpoints/local.pt

# Extract state_dict from PyTorch Lightning
python scripts/manage_checkpoints.py convert \
    --input data/checkpoints/lightning.ckpt \
    --output data/checkpoints/state_dict.pt \
    --extract-lightning
```

### Compare Checkpoints

```bash
# Compare two checkpoints
python scripts/manage_checkpoints.py compare \
    --checkpoint1 model_v1.ckpt \
    --checkpoint2 model_v2.ckpt

# Compare weight values (slower)
python scripts/manage_checkpoints.py compare \
    --checkpoint1 model_v1.ckpt \
    --checkpoint2 model_v2.ckpt \
    --compare-weights
```

### Test Loading

```bash
# Load and test checkpoint
python scripts/manage_checkpoints.py load \
    --checkpoint data/checkpoints/clipzyme_model.ckpt \
    --device cuda \
    --test-inference

# Load pretrained model by name
python scripts/manage_checkpoints.py load \
    --pretrained clipzyme \
    --device cuda \
    --download \
    --test-inference
```

### List Available Checkpoints

```bash
python scripts/manage_checkpoints.py list --cache-dir data/checkpoints
```

## 🐍 Python API

### ZenodoDownloader

```python
from checkpoints import ZenodoDownloader

downloader = ZenodoDownloader(output_dir="data/checkpoints")

# Download specific file
model_path = downloader.download_clipzyme_file('model', extract=True)

# Download all files
paths = downloader.download_all(extract=True)

# List available files
files = downloader.list_files()
```

### CheckpointLoader

```python
from checkpoints import CheckpointLoader

loader = CheckpointLoader(device="cuda")

# Load any checkpoint format
model = loader.load("data/checkpoints/clipzyme_model.ckpt")

# With custom config
from config import CLIPZymeConfig
config = CLIPZymeConfig.get_preset('clipzyme_faithful')
model = loader.load("checkpoint.ckpt", config=config)
```

### StateDictConverter

```python
from checkpoints import StateDictConverter
import torch

converter = StateDictConverter()

# Load official checkpoint
official_state_dict = torch.load("official.ckpt")['state_dict']

# Convert to local format
local_state_dict = converter.convert(
    official_state_dict,
    source_format="official",
    target_format="local"
)

# Analyze differences
differences = converter.analyze_differences(
    state_dict1,
    state_dict2,
    verbose=True
)

# Suggest parameter mappings
mappings = converter.suggest_mappings(official_state_dict, local_state_dict)
```

### Validation

```python
from checkpoints import validate_checkpoint, compare_checkpoints

# Validate checkpoint
is_valid = validate_checkpoint(
    "data/checkpoints/clipzyme_model.ckpt",
    expected_params=654987321,
    test_forward=True
)

# Compare checkpoints
results = compare_checkpoints(
    "checkpoint1.ckpt",
    "checkpoint2.ckpt",
    compare_weights=True
)

print(f"Keys match: {results['keys_match']}")
print(f"Shapes match: {results['shapes_match']}")
print(f"Max weight diff: {results['max_weight_diff']:.2e}")
```

## 📁 Checkpoint Formats Supported

### PyTorch Lightning (Official)

```python
{
    'epoch': 29,
    'global_step': 45600,
    'state_dict': {...},  # Model parameters
    'optimizer_states': [...],
    'lr_schedulers': [...],
    'hyper_parameters': {...}
}
```

### Standard PyTorch State Dict

```python
{
    'protein_encoder.esm_model.weight': tensor(...),
    'reaction_encoder.dmpnn.weight': tensor(...),
    ...
}
```

### Full Model

```python
# torch.save(model, path)
CLIPZymeModel(...)
```

### Pickle

```python
# pickle.dump(model, f)
CLIPZymeModel(...) or {'model': CLIPZymeModel(...)}
```

## 🔄 Parameter Name Mapping

The converter automatically handles different naming conventions:

| Official Format | Local Format |
|----------------|--------------|
| `model.protein_encoder.*` | `protein_encoder.*` |
| `model.reaction_encoder.*` | `reaction_encoder.*` |
| `esm.*` | `protein_encoder.esm_model.*` |
| `dmpnn.*` | `reaction_encoder.*` |
| `prot_projection.*` | `protein_encoder.projection.*` |
| `logit_scale` | `temperature` |

Custom mappings can be added to `StateDictConverter.PARAMETER_MAPPINGS`.

## 📊 Zenodo Files

CLIPZyme Official Repository: `https://zenodo.org/records/15161343`

| File | Size | Description |
|------|------|-------------|
| `clipzyme_model.zip` | 2.4 GB | Pre-trained model checkpoint |
| `clipzyme_data.zip` | 1.3 GB | Training and evaluation datasets |
| `reaction_rule_split.p` | 1.9 kB | Train/test splits |

**Total:** 3.7 GB

## 🎯 Use Cases

### 1. Load Official Pretrained Model

```python
from models import load_pretrained

# Simple one-liner
model = load_pretrained("clipzyme", device="cuda")

# Use for inference
reaction_smiles = ["[C:1]=[O:2]>>[C:1]-[O:2]"]
embeddings = model.encode_reactions(reaction_smiles)
```

### 2. Fine-tune Official Model

```python
from models import load_pretrained

# Load pretrained model
model = load_pretrained("clipzyme", device="cuda")

# Unfreeze for fine-tuning
for param in model.parameters():
    param.requires_grad = True

# Fine-tune on your data
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# ... training loop
```

### 3. Compare Training Runs

```python
from checkpoints import compare_checkpoints

# Compare checkpoints from different epochs
results = compare_checkpoints(
    "checkpoints/epoch_10.ckpt",
    "checkpoints/epoch_20.ckpt",
    compare_weights=True
)

print(f"Parameter changes: {results['max_weight_diff']:.2e}")
```

### 4. Build Screening Set with Official Model

```python
from models import load_pretrained
from screening import build_screening_set_from_model, ProteinDatabase

# Load official model
model = load_pretrained("clipzyme", device="cuda")

# Load your proteins
protein_db = ProteinDatabase()
protein_db.load_from_csv("my_proteins.csv")

# Build screening set
screening_set = build_screening_set_from_model(
    model=model,
    protein_database=protein_db,
    batch_size=32,
    device="cuda"
)

# Save for later
screening_set.save_to_pickle("my_screening_set.p")
```

### 5. Checkpoint Surgery

```python
import torch
from checkpoints import StateDictConverter

# Load checkpoint
checkpoint = torch.load("official.ckpt")
state_dict = checkpoint['state_dict']

# Modify specific parameters
state_dict['temperature'] = torch.tensor(0.05)  # Change temperature

# Remove specific layers
keys_to_remove = [k for k in state_dict.keys() if 'projection' in k]
for key in keys_to_remove:
    del state_dict[key]

# Save modified checkpoint
torch.save({'state_dict': state_dict}, "modified.ckpt")
```

## 🔍 Troubleshooting

### Download Issues

```bash
# Check if Zenodo is accessible
curl -I https://zenodo.org/records/15161343

# Try manual download and place in data/checkpoints/
# Then use: --no-extract flag if needed
```

### Loading Errors

```python
# Inspect checkpoint first
from checkpoints import inspect_checkpoint
info = inspect_checkpoint("problematic.ckpt")

# Try with strict=False
from checkpoints import CheckpointLoader
loader = CheckpointLoader()
model = loader.load("problematic.ckpt", strict=False)
```

### Parameter Mismatches

```python
# Analyze differences
from checkpoints import StateDictConverter
import torch

official = torch.load("official.ckpt")['state_dict']
local = torch.load("local.pt")

converter = StateDictConverter()
differences = converter.analyze_differences(official, local, verbose=True)

# Suggest mappings
mappings = converter.suggest_mappings(official, local)
```

### Out of Memory

```bash
# Download checkpoints to disk first, don't keep in memory
python scripts/manage_checkpoints.py download --extract

# Load with CPU, then move to GPU layer by layer
python scripts/manage_checkpoints.py load \
    --checkpoint checkpoint.ckpt \
    --device cpu
```

## 📝 Best Practices

1. **Cache Downloads**: Keep checkpoints in `data/checkpoints/` to avoid re-downloading
2. **Validate After Download**: Always validate checkpoints after downloading
3. **Use load_pretrained()**: Easiest way to get started with official models
4. **Inspect Unknown Checkpoints**: Use `inspect` command before loading
5. **Convert for Compatibility**: Convert official checkpoints to local format if needed
6. **Test Inference**: Always test inference after loading

## 🔗 Related Documentation

- [Model Architecture](../models/README.md) - Model implementation details
- [Training Guide](../TRAINING.md) - How to train models
- [Screening System](../screening/README.md) - Using models for screening

## 💡 Tips

- **First Time**: Use `load_pretrained()` - it handles everything automatically
- **Offline Mode**: Download checkpoints once, then disable `download_if_missing`
- **Custom Models**: Train your own and save with `model.save_pretrained()`
- **Version Control**: Track checkpoint hashes, not the files themselves
- **Sharing**: Share converted `.pt` files instead of full Lightning checkpoints

---

For examples, see `scripts/demo_checkpoints.py`

For issues, see GitHub Issues: https://github.com/anthropics/claude-code/issues
