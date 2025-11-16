# Getting Started with CLIPZyme

Quick guide to start using CLIPZyme for enzyme-reaction virtual screening.

---

## 🎯 What CLIPZyme Does

CLIPZyme performs **bidirectional virtual screening**:

### Option A: Reaction → Proteins
```
Input:  Chemical reaction SMILES
Output: Ranked list of proteins that can catalyze it

Example:
  Reaction: [CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]
  Results:
    1. P80222 (Alcohol Dehydrogenase) - Score: 0.95
    2. P00325 (NAD-dependent oxidase)  - Score: 0.87
    3. ...
```

### Option B: Protein → Reactions
```
Input:  Protein sequence
Output: Ranked list of reactions it can catalyze

Example:
  Protein: MSTAGKVIKCKAAVLWEEKKPFS...
  Results:
    1. Ethanol → Acetaldehyde   - Score: 0.92
    2. Methanol → Formaldehyde  - Score: 0.81
    3. ...
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Test Architecture (No Download Required)

Test that everything works with a fresh model:

```bash
python test_architecture.py
```

This will:
- ✅ Create a CLIPZyme model from scratch
- ✅ Test reaction encoding
- ✅ Test protein encoding
- ✅ Compute similarity scores
- ⚠️ Uses random weights (untrained model)

**Note:** This downloads ESM2-650M (~2.5 GB) on first run.

---

### Step 2: Download Pretrained Model (Recommended)

Get the official pretrained checkpoint:

```bash
python quick_start.py
```

This interactive script will:
1. Download the pretrained checkpoint (~2.4 GB from Zenodo)
2. Load the model
3. Run a screening demo

**Manual Download:**
If automatic download fails:
1. Visit: https://zenodo.org/records/15161343
2. Download: `clipzyme_model.zip` (2.4 GB)
3. Extract to: `data/checkpoints/`

---

### Step 3: Run Virtual Screening

Once you have the pretrained model:

```python
from models import load_checkpoint
from inference import CLIPZymePredictor

# Load pretrained model
model = load_checkpoint("data/checkpoints/clipzyme_model.ckpt")

# Create predictor
predictor = CLIPZymePredictor(model)

# Screen a reaction
reaction = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
result = predictor.screen(reaction, top_k=100)

# Get top proteins
for protein, score in zip(result.top_proteins[:10], result.scores[:10]):
    print(f"{protein}: {score:.4f}")
```

---

## 📚 Example Scripts

### Demo Scripts (in `scripts/`)

```bash
# Virtual screening demo
python scripts/demo_screening.py

# Inference API demo
python scripts/demo_inference.py

# Training demo (requires training data)
python scripts/demo_training.py

# Checkpoint management
python scripts/demo_checkpoints.py

# Evaluation metrics
python scripts/demo_evaluation.py
```

---

## 💾 What You Need

### For Using Pretrained Model:
- ✅ Pretrained checkpoint (~2.4 GB) - Download from Zenodo
- ✅ Screening set (260K proteins) - Optional, for large-scale screening
- ✅ Your reaction SMILES or protein sequences

### For Training from Scratch:
- ⚠️ Training dataset (~1.3 GB) - Download from Zenodo
- ⚠️ GPU with 24GB+ VRAM (e.g., A100)
- ⚠️ 1-2 weeks training time

---

## 📖 Usage Examples

### Example 1: Screen One Reaction

```python
from models import load_checkpoint

# Load model
model = load_checkpoint("data/checkpoints/clipzyme_model.ckpt")
model.eval()

# Encode reaction
reaction = "[C:1]=[O:2]>>[C:1]-[O:2]"
reaction_emb = model.encode_reactions([reaction])[0]

# Encode proteins
proteins = ["MSTAGKVIK...", "MKKFVLIGL..."]
protein_embs = model.encode_proteins(proteins)

# Compute similarities
import torch
scores = torch.cosine_similarity(
    reaction_emb.unsqueeze(0),
    protein_embs,
    dim=1
)

# Get top matches
top_indices = torch.argsort(scores, descending=True)[:10]
for i, idx in enumerate(top_indices, 1):
    print(f"{i}. Protein {idx}: {scores[idx]:.4f}")
```

### Example 2: Batch Screening

```python
from inference import CLIPZymePredictor

# Load predictor
predictor = CLIPZymePredictor.from_checkpoint("data/checkpoints/clipzyme_model.ckpt")

# Batch screen multiple reactions
reactions = [
    "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]",
    "[C:1]=[O:2]>>[C:1]-[O:2]",
    # ... more reactions
]

results = predictor.screen_batch(reactions, top_k=100)

for i, result in enumerate(results):
    print(f"\nReaction {i+1}:")
    print(f"Top protein: {result.top_proteins[0]}")
    print(f"Score: {result.scores[0]:.4f}")
```

### Example 3: Using the Screening Set

```python
from screening.screening_set import ScreeningSet

# Load pre-embedded screening set (260K proteins)
screening_set = ScreeningSet.load_from_pickle("data/screening_set.pkl")

# Query with reaction embedding
reaction_emb = model.encode_reactions([reaction])[0]

# Find top matches
scores, indices, protein_ids = screening_set.compute_similarity(
    reaction_emb,
    top_k=100,
    return_scores=True
)

print(f"Found {len(protein_ids)} matches")
for i, (pid, score) in enumerate(zip(protein_ids[:10], scores[:10]), 1):
    print(f"{i}. {pid}: {score:.4f}")
```

---

## 🔧 Configuration

The default configuration (`configs/default.yaml`) matches the paper exactly:

```yaml
protein_encoder:
  type: EGNN                              # ESM2 + E(n)-GNN
  plm_name: facebook/esm2_t33_650M_UR50D  # 650M parameters
  egnn_layers: 6
  egnn_hidden_dim: 1280
  k_neighbors: 30
  distance_cutoff: 10.0

reaction_encoder:
  type: DMPNN                             # Two-Stage DMPNN
  feature_type: clipzyme                  # 9 atom + 3 edge features
  dmpnn_hidden_dim: 1280
  num_layers: 5

training:
  temperature: 0.07
  learning_rate: 0.0001
  batch_size: 64
```

---

## ⚠️ Important Notes

### Reaction SMILES Requirements:
- **Must have atom mapping**: `[C:1]=[O:2]>>[C:1]-[O:2]` ✅
- **Without mapping will fail**: `C=O>>C-O` ❌

### Protein Requirements:
- For full accuracy, need **3D structures** (from AlphaFold)
- Can use sequence-only (ESM2 embeddings) with reduced accuracy

### Hardware:
- **Inference**: CPU is fine (slower) or GPU
- **Training**: Requires GPU with 24GB+ VRAM

---

## 🐛 Troubleshooting

### "Checkpoint not found"
```bash
# Download manually
python quick_start.py
```

### "ESM2 model downloading"
```
This is normal on first run. ESM2-650M is ~2.5 GB.
Download happens once, then cached.
```

### "Out of memory"
```python
# Use smaller batch size
predictor = CLIPZymePredictor(model, config=PredictorConfig(batch_size=8))
```

### "Reaction parsing failed"
```
Make sure reaction SMILES has atom mapping:
  [C:1]=[O:2]>>[C:1]-[O:2]  ✅
  C=O>>C-O                   ❌
```

---

## 📚 More Resources

- **README.md** - Project overview
- **docs/SCREENING_SYSTEM.md** - Detailed screening guide
- **docs/CHECKPOINTS_INTEGRATION.md** - Checkpoint management
- **docs/EVALUATION_SYSTEM.md** - Evaluation metrics
- **docs/PROJECT_STATUS.md** - Implementation status

---

## 🤝 Support

- **Issues**: https://github.com/Vakyro/project/issues
- **Paper**: https://arxiv.org/abs/2402.06748
- **Zenodo**: https://zenodo.org/records/15161343

---

Happy enzyme screening! 🧬
