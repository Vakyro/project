# CLIPZyme Screening System

Complete virtual screening system for matching enzymes to chemical reactions.

## 📋 Overview

The screening system enables you to:
- Screen single reactions against 100K+ proteins (interactive mode)
- High-throughput screening of thousands of reactions (batched mode)
- Build custom screening sets from protein databases
- Evaluate results with BEDROC and other metrics
- Cache embeddings for faster repeated screening

## 🚀 Quick Start

### 1. Interactive Screening (Single Reaction)

```python
from screening import InteractiveScreener, ScreeningSet
from models.clipzyme import CLIPZymeModel

# Load model and screening set
model = CLIPZymeModel.from_pretrained("models/clipzyme.pt")
screening_set = ScreeningSet().load_from_pickle("data/screening_set.p")

# Create screener
screener = InteractiveScreener(model, screening_set)

# Screen a reaction
result = screener.screen_reaction(
    "[C:1]=[O:2]>>[C:1]-[O:2]",
    top_k=100
)

print(f"Top match: {result.ranked_protein_ids[0]}")
print(f"Score: {result.scores[0]:.4f}")
```

### 2. Batched Screening (High-Throughput)

```python
from screening import BatchedScreener, BatchedScreeningConfig

# Configure batched screener
config = BatchedScreeningConfig(
    batch_size=64,
    devices=["cuda:0", "cuda:1"],  # Multi-GPU
    top_k=100
)

screener = BatchedScreener(model, screening_set, config=config)

# Screen from CSV
results = screener.screen_from_csv(
    csv_path="reactions.csv",
    reaction_column="reaction_smiles",
    show_progress=True
)

print(f"Screened {len(results)} reactions")
```

### 3. Build Screening Set

```bash
# Encode proteins and create screening set
python scripts/build_screening_set.py --config configs/build_screening_set.yaml
```

## 📚 Components

### ScreeningSet

Manages collections of pre-embedded proteins.

```python
from screening import ScreeningSet

# Load pre-computed embeddings
screening_set = ScreeningSet(embedding_dim=512, device="cuda")
screening_set.load_from_pickle("data/screening_set.p")

print(f"Loaded {len(screening_set)} proteins")

# Get embedding for specific protein
embedding = screening_set.get_embedding("P12345")

# Compute similarity with reaction
scores, indices, protein_ids = screening_set.compute_similarity(
    reaction_embedding,
    top_k=100
)
```

### InteractiveScreener

Memory-efficient screening for exploratory analysis.

```python
from screening import InteractiveScreener, InteractiveScreeningConfig

config = InteractiveScreeningConfig(
    top_k=100,
    device="cuda",
    evaluate=True  # Compute metrics if true positives available
)

screener = InteractiveScreener(model, screening_set, config=config)

# Screen single reaction
result = screener.screen_reaction(
    reaction_smiles="[C:1]=[O:2]>>[C:1]-[O:2]",
    reaction_id="rxn_001",
    true_positives=["P12345", "P67890"]  # For evaluation
)

# Screen multiple reactions
results = screener.screen_reactions(
    reaction_smiles_list=["rxn1", "rxn2", "rxn3"],
    show_progress=True
)

# Compare two reactions
comparison = screener.compare_reactions(rxn1, rxn2, top_k=10)
print(f"Overlap: {comparison['overlap_count']} proteins")
print(f"Reaction similarity: {comparison['reaction_similarity']:.3f}")
```

### BatchedScreener

High-performance screening with multi-GPU support.

```python
from screening import BatchedScreener, BatchedScreeningConfig

config = BatchedScreeningConfig(
    batch_size=64,
    num_workers=4,
    top_k=100,
    devices=["cuda:0", "cuda:1"],
    output_dir="results/screening",
    save_scores=True,
    evaluate=True
)

screener = BatchedScreener(model, screening_set, config=config)

# Screen from CSV
results = screener.screen_from_csv(
    csv_path="reactions.csv",
    reaction_column="reaction_smiles",
    id_column="reaction_id",
    true_positives_column="known_enzymes"  # Comma-separated
)

# Results are automatically saved to output_dir
```

### Ranking Metrics

Evaluate screening performance with standard metrics.

```python
from screening.ranking import (
    compute_bedroc,
    evaluate_screening,
    batch_evaluate_screening
)

# Compute BEDROC (primary metric in CLIPZyme paper)
bedroc = compute_bedroc(
    scores=[0.9, 0.8, 0.7, 0.3, 0.2],
    labels=[1, 1, 0, 1, 0],
    alpha=20.0  # CLIPZyme uses α=20
)
print(f"BEDROC: {bedroc:.3f}")

# Comprehensive evaluation
metrics = evaluate_screening(
    ranked_protein_ids=result.ranked_protein_ids,
    scores=result.scores,
    true_positives=["P12345", "P67890"]
)

print(f"BEDROC_20: {metrics['BEDROC_20']:.3f}")
print(f"Top10 Accuracy: {metrics['Top10_Accuracy']:.3f}")
print(f"EF_1%: {metrics['EF_1%']:.2f}")

# Aggregate metrics across multiple reactions
aggregate = batch_evaluate_screening(results_list)
```

### EmbeddingCache

Cache embeddings for faster repeated screening.

```python
from screening.cache import EmbeddingCache

# Two-level cache (memory + disk)
cache = EmbeddingCache(
    memory_cache_size=1000,
    disk_cache_dir="cache/embeddings",
    disk_cache_size_gb=5.0
)

# Get cached embedding
embedding = cache.get("protein_id", device="cuda")

# Cache new embedding
cache.put("protein_id", embedding)

# Batch operations
cached_keys, cached_embs, missing_keys = cache.get_batch(
    keys=["P1", "P2", "P3"],
    device="cuda"
)

# Statistics
stats = cache.get_stats()
print(f"Memory cache hit rate: {stats['memory']['hit_rate']:.2%}")
```

## 📊 Metrics

### BEDROC (Boltzmann-Enhanced Discrimination of ROC)

Primary metric in CLIPZyme paper. Emphasizes early recognition.

- **BEDROC_20**: Standard CLIPZyme metric (α=20)
- **BEDROC_50**: More emphasis on early hits
- **BEDROC_85**: Even stronger early emphasis

**Interpretation**: Higher is better, range [0, 1]

### Top-K Accuracy

Fraction of true positives found in top-K predictions.

```python
Top1_Accuracy   # Did we get #1 rank correct?
Top5_Accuracy   # Are any true positives in top 5?
Top10_Accuracy
Top50_Accuracy
Top100_Accuracy
```

### Enrichment Factor (EF)

How much better than random ranking.

```python
EF_1%   # Enrichment in top 1% of predictions
EF_5%   # Enrichment in top 5%
EF_10%  # Enrichment in top 10%
```

**Interpretation**: EF > 1 means better than random

### Other Metrics

- **AUROC**: Area under ROC curve
- **Average Precision**: Area under precision-recall curve

## 🔧 Configuration Files

### Interactive Mode (`configs/screening_interactive.yaml`)

```yaml
screening:
  mode: "interactive"
  top_k: 100
  evaluate: true

model:
  checkpoint_path: "models/clipzyme_model.pt"
  device: "cuda"

screening_set:
  embeddings_path: "data/screening_set.p"
  sequences_path: "data/uniprot2sequence.p"

output:
  output_dir: "results/screening"
  save_rankings: true
```

### Batched Mode (`configs/screening_batched.yaml`)

```yaml
screening:
  mode: "batched"
  batch_size: 64
  num_workers: 4
  top_k: 100

model:
  devices: ["cuda:0", "cuda:1"]  # Multi-GPU

reactions:
  csv_path: "data/reactions.csv"
  reaction_column: "reaction_smiles"

output:
  output_dir: "results/screening_batch"
  save_intermediate: true
```

## 💻 Command Line Usage

### Run Screening

```bash
# Interactive mode
python scripts/run_screening.py --config configs/screening_interactive.yaml

# Batched mode
python scripts/run_screening.py --config configs/screening_batched.yaml

# Quick test (single reaction)
python scripts/run_screening.py \
    --reaction "[C:1]=[O:2]>>[C:1]-[O:2]" \
    --model models/clipzyme.pt \
    --screening-set data/screening_set.p \
    --top-k 10
```

### Build Screening Set

```bash
# Build from CSV
python scripts/build_screening_set.py --config configs/build_screening_set.yaml

# Custom output path
python scripts/build_screening_set.py \
    --config configs/build_screening_set.yaml \
    --output data/my_screening_set.p
```

## 📈 Performance Optimization

### Multi-GPU Screening

```python
config = BatchedScreeningConfig(
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    batch_size=64  # Per GPU
)
```

### Mixed Precision

```yaml
optimization:
  use_amp: true  # Automatic Mixed Precision
  pin_memory: true
```

### Caching

```python
# Enable caching for repeated screening
cache = EmbeddingCache(
    memory_cache_size=5000,  # 5K proteins in RAM
    disk_cache_dir="cache/embeddings",
    disk_cache_size_gb=10.0
)

screener = InteractiveScreener(model, screening_set)
# Cache is used automatically
```

## 📁 File Formats

### Screening Set (screening_set.p)

Pickle file with Dict[str, Tensor]:
```python
{
    "P12345": tensor([0.1, 0.2, ..., 0.9]),  # 512D embedding
    "P67890": tensor([0.3, 0.4, ..., 0.8]),
    ...
}
```

### Protein Database (uniprot2sequence.p)

Pickle file with Dict[str, str]:
```python
{
    "P12345": "MKVLWAALLVTFLAGCQAKV...",
    "P67890": "MTMDKSELVQKAKLAEQAER...",
    ...
}
```

### Reactions CSV

```csv
reaction_id,reaction_smiles,known_enzymes
rxn_001,[C:1]=[O:2]>>[C:1]-[O:2],"P12345,P67890"
rxn_002,[N:1]=[N:2]>>[N:1]-[N:2],"P11111"
```

### Output Rankings JSON

```json
{
  "reaction_id": "rxn_001",
  "top_proteins": ["P12345", "P67890", ...],
  "scores": [0.95, 0.89, ...],
  "metrics": {
    "BEDROC_20": 0.856,
    "Top10_Accuracy": 1.0,
    "EF_1%": 45.2
  }
}
```

## 🎯 Use Cases

### 1. Find Enzymes for Novel Reaction

```python
screener = InteractiveScreener(model, screening_set)

result = screener.screen_reaction(
    "[new_reaction_smiles]",
    top_k=50
)

# Investigate top candidates
for protein_id in result.ranked_protein_ids[:10]:
    info = screener.get_protein_info(protein_id)
    print(f"{protein_id}: {info['sequence'][:50]}...")
```

### 2. Benchmark on Test Set

```python
screener = BatchedScreener(model, screening_set)

results = screener.screen_from_csv(
    "benchmark/test_reactions.csv",
    reaction_column="reaction_smiles",
    true_positives_column="known_enzymes"
)

metrics = batch_evaluate_screening(results)
print(f"BEDROC_20: {metrics['BEDROC_20']:.3f}")
```

### 3. Build Custom Screening Set

```python
# Load your proteins
protein_db = ProteinDatabase()
protein_db.load_from_csv(
    "my_proteins.csv",
    id_column="protein_id",
    sequence_column="sequence"
)

# Encode with model
screening_set = build_screening_set_from_model(
    model=model,
    protein_database=protein_db,
    batch_size=32,
    device="cuda"
)

# Save for later
screening_set.save_to_pickle("my_screening_set.p")
```

## 🔬 CLIPZyme Paper Reproduction

The official CLIPZyme paper uses:
- **261,907 enzymes** from UniProt
- **BEDROC_85** as primary metric
- **Top-1 accuracy** for ranking quality
- **ESM2-T33 (650M) + EGNN** protein encoder
- **Two-stage DMPNN** reaction encoder

Reproduce with:
```bash
python scripts/run_screening.py --config configs/screening_batched.yaml
```

Ensure `clipzyme_faithful.yaml` is used for model architecture.

## 📖 References

- **BEDROC**: Truchon & Bayly (2007) "Evaluating Virtual Screening Methods"
- **CLIPZyme**: Mikhael et al. (2024) "Reaction-Conditioned Virtual Screening of Enzymes"

## 🐛 Troubleshooting

### Out of Memory

```python
# Reduce batch size
config.batch_size = 32

# Use CPU for screening set
screening_set = ScreeningSet(device="cpu")

# Enable gradient checkpointing (if building screening set)
```

### Slow Screening

```python
# Use multi-GPU
config.devices = ["cuda:0", "cuda:1"]

# Increase batch size
config.batch_size = 128

# Enable caching
cache = EmbeddingCache(memory_cache_size=10000)
```

### Cache Issues

```python
# Clear cache
cache.clear()

# Check stats
print(cache.get_stats())
```

## 💡 Tips

1. **Pre-compute screening sets** for databases you use frequently
2. **Use batched mode** for >100 reactions
3. **Enable caching** for iterative screening
4. **Multi-GPU** scales almost linearly for batch mode
5. **BEDROC_20** is the standard metric, but explore others
6. **Filter proteins** by length before building screening set to save memory

---

For more examples, see `scripts/demo_screening.py`
