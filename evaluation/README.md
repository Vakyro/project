# CLIPZyme Evaluation System

Complete evaluation system with BEDROC₈₅ and all metrics from the CLIPZyme paper.

## 📋 Overview

The evaluation system implements all metrics used in the CLIPZyme paper (Mikhael et al. 2024):

- **BEDROC₈₅**: Primary metric (α=85 for strong early recognition)
- **BEDROC₅₀, BEDROC₂₀**: Additional BEDROC variants
- **Top-K Accuracy**: Hit rate in top K predictions
- **Enrichment Factor**: Enrichment at 1%, 5%, 10%
- **AUROC**: Area under ROC curve
- **AUPRC**: Average precision (area under PR curve)
- **Statistical Analysis**: Bootstrap, significance testing

## 🎯 CLIPZyme Paper Metrics

### Primary Metric: BEDROC₈₅

The paper reports:
- **BEDROC₈₅ = 44.69%** (without EC information)
- **BEDROC₈₅ = 75.57%** (with EC2 prediction)

BEDROC (Boltzmann-Enhanced Discrimination of ROC) emphasizes early recognition:
- α=20: Standard, moderate early emphasis
- α=50: Stronger early emphasis
- α=85: Very strong early emphasis (CLIPZyme)

## 🚀 Quick Start

### Compute All Metrics

```python
from evaluation import compute_all_metrics

metrics = compute_all_metrics(
    ranked_ids=result.ranked_protein_ids,
    scores=result.scores,
    active_ids=["P12345", "P67890"]  # Known active enzymes
)

print(f"BEDROC_85: {metrics.bedroc_85:.4f}")
print(f"Top-1 Accuracy: {metrics.top1_accuracy:.4f}")
print(f"AUROC: {metrics.auroc:.4f}")
```

### Run Benchmark Evaluation

```python
from models import load_pretrained
from screening import ScreeningSet
from evaluation import run_benchmark

# Load model and data
model = load_pretrained("clipzyme", device="cuda")
screening_set = ScreeningSet().load_from_pickle("screening_set.p")

# Run benchmark
results = run_benchmark(
    model=model,
    screening_set=screening_set,
    test_reactions=test_reactions,
    true_labels=true_labels,
    output_dir="results/benchmark",
    compare_to_paper=True
)

print(f"BEDROC_85: {results['aggregated_metrics'].bedroc_85:.4f}")
```

### Command Line

```bash
# Run complete evaluation
python scripts/run_evaluation.py \
    --model clipzyme \
    --screening-set data/screening_set.p \
    --test-data data/test_reactions.csv \
    --compare-to-paper \
    --bootstrap \
    --output results/evaluation
```

## 📊 Metrics Explained

### BEDROC (Boltzmann-Enhanced Discrimination of ROC)

Measures early recognition performance with exponential weighting.

```python
from evaluation import CLIPZymeMetrics

bedroc_85 = CLIPZymeMetrics.compute_bedroc(
    scores=prediction_scores,
    labels=true_labels,
    alpha=85.0  # CLIPZyme paper uses α=85
)
```

**Interpretation**:
- 0.0: Random performance
- 1.0: Perfect early recognition
- CLIPZyme paper: 44.69% (baseline), 75.57% (with EC2)

**Alpha values**:
- α=20: Moderate emphasis on early hits (standard)
- α=50: Strong emphasis on early hits
- α=85: Very strong emphasis on early hits (CLIPZyme)

### Top-K Accuracy

Fraction of true positives found in top K predictions.

```python
top10_acc = CLIPZymeMetrics.compute_top_k_accuracy(
    ranked_ids=ranked_protein_ids,
    active_ids=known_active_enzymes,
    k=10
)
```

**Example**: Top-10 accuracy of 0.60 means 60% of known active enzymes are in the top 10 predictions.

### Enrichment Factor

How much better than random ranking.

```python
ef_1pct = CLIPZymeMetrics.compute_enrichment_factor(
    ranked_ids=ranked_protein_ids,
    active_ids=known_active_enzymes,
    fraction=0.01  # Top 1% of database
)
```

**Interpretation**:
- EF = 1.0: Random performance
- EF > 1.0: Better than random
- EF = 10: 10× enrichment vs random

**Example**: EF 1% = 45 means the top 1% contains 45× more actives than expected by random.

### AUROC (Area Under ROC Curve)

Standard classification metric.

```python
auroc = CLIPZymeMetrics.compute_auroc(
    scores=prediction_scores,
    labels=true_labels
)
```

**Interpretation**:
- 0.5: Random
- 1.0: Perfect
- >0.7: Good
- >0.9: Excellent

### AUPRC (Average Precision)

Area under Precision-Recall curve, better for imbalanced datasets.

```python
auprc = CLIPZymeMetrics.compute_average_precision(
    scores=prediction_scores,
    labels=true_labels
)
```

## 📈 Visualization

### ROC Curve

```python
from evaluation import plot_roc_curve

plot_roc_curve(
    scores=prediction_scores,
    labels=true_labels,
    title="ROC Curve - CLIPZyme",
    save_path="roc_curve.png"
)
```

### Precision-Recall Curve

```python
from evaluation import plot_pr_curve

plot_pr_curve(
    scores=prediction_scores,
    labels=true_labels,
    title="Precision-Recall Curve",
    save_path="pr_curve.png"
)
```

### BEDROC Comparison

```python
from evaluation import plot_bedroc_comparison

metrics_dict = {
    'Baseline': baseline_metrics,
    'Improved': improved_metrics,
    'Paper': paper_metrics
}

plot_bedroc_comparison(
    metrics_dict,
    title="BEDROC Comparison",
    save_path="bedroc_comparison.png"
)
```

### Complete Evaluation Report

```python
from evaluation import create_evaluation_report

# Creates all plots + metrics text file
report_files = create_evaluation_report(
    metrics=metrics,
    scores=scores,
    labels=labels,
    output_dir="results/evaluation",
    name="my_evaluation"
)
```

Generates:
- `my_evaluation_roc_curve.png`
- `my_evaluation_pr_curve.png`
- `my_evaluation_summary.png`
- `my_evaluation_metrics.txt`

## 📊 Statistical Analysis

### Bootstrap Confidence Intervals

```python
from evaluation import bootstrap_metrics

ci = bootstrap_metrics(
    ranked_ids_list=all_ranked_ids,
    scores_list=all_scores,
    active_ids_list=all_active_ids,
    n_bootstrap=1000,
    metric_name='bedroc_85'
)

print(f"BEDROC_85: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
```

### Significance Testing

```python
from evaluation import significance_test

# Compare two models
result = significance_test(
    metrics_a=baseline_metrics_list,
    metrics_b=improved_metrics_list,
    metric_name='bedroc_85',
    test_type='paired_t'
)

print(f"p-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant']}")
print(f"Effect size (Cohen's d): {result['cohens_d']:.2f}")
```

### Effect Size

```python
from evaluation import compute_effect_size

cohens_d = compute_effect_size(
    metrics_a=baseline_metrics,
    metrics_b=improved_metrics,
    metric_name='bedroc_85'
)

# Interpretation:
# d = 0.2: Small effect
# d = 0.5: Medium effect
# d = 0.8: Large effect
```

## 🎯 Benchmark Against Paper

### Compare to CLIPZyme Paper

```python
from evaluation import compare_to_paper_results

compare_to_paper_results(your_metrics)
```

Output:
```
COMPARISON TO CLIPZYME PAPER RESULTS
======================================================================

📊 Our Results:
  BEDROC_85: 0.4650
  BEDROC_50: 0.4210
  BEDROC_20: 0.3890
  Top-1 Acc: 0.2600
  AUROC:     0.8450

📄 Paper Results (Baseline):
  BEDROC_85: 0.4469
  Dataset: EnzymeMap

📄 Paper Results (With EC2):
  BEDROC_85: 0.7557

📈 Comparison:
  vs Baseline: +0.0181 (+4.1%)
  vs With EC2: -0.2907 (-38.5%)
```

### Benchmark Evaluator

```python
from evaluation import BenchmarkEvaluator

evaluator = BenchmarkEvaluator(model, screening_set)

# Run benchmark
results = evaluator.run_benchmark(
    test_reactions=test_reactions,
    true_labels=true_labels,
    top_k=100
)

# Compare to paper
comparison = evaluator.compare_to_paper(results, use_ec=False)
```

### Paper Results Reference

From Mikhael et al. (2024):

| Method | BEDROC₈₅ | Dataset | Screening Set |
|--------|----------|---------|---------------|
| CLIPZyme (baseline) | 44.69% | EnzymeMap | 260,197 enzymes |
| CLIPZyme + EC2 | 75.57% | EnzymeMap | 260,197 enzymes |

## 💻 Command Line Usage

### Basic Evaluation

```bash
python scripts/run_evaluation.py \
    --model data/checkpoints/clipzyme_model.ckpt \
    --screening-set data/screening_set.p \
    --test-data data/test_reactions.csv \
    --output results/evaluation
```

### With Bootstrap and Paper Comparison

```bash
python scripts/run_evaluation.py \
    --model clipzyme \
    --screening-set data/screening_set.p \
    --test-data data/test_reactions.csv \
    --compare-to-paper \
    --bootstrap \
    --n-bootstrap 1000 \
    --output results/evaluation
```

### Specify Data Columns

```bash
python scripts/run_evaluation.py \
    --model clipzyme \
    --screening-set data/screening_set.p \
    --test-data data/test_reactions.csv \
    --reaction-column "rxn_smiles" \
    --id-column "rxn_id" \
    --labels-column "enzymes" \
    --output results/evaluation
```

## 📁 Output Files

After running evaluation, you'll get:

```
results/evaluation/
├── clipzyme_evaluation_roc_curve.png       # ROC curve
├── clipzyme_evaluation_pr_curve.png        # PR curve
├── clipzyme_evaluation_summary.png         # All metrics summary
├── clipzyme_evaluation_metrics.txt         # Text report
├── benchmark_metrics.json                  # JSON metrics
└── paper_comparison.json                   # Comparison to paper
```

## 🔬 Advanced Usage

### Stratify by EC Class

```python
from evaluation import BenchmarkEvaluator

evaluator = BenchmarkEvaluator(model, screening_set)

# Evaluate by EC class
ec_results = evaluator.evaluate_by_ec_class(
    test_reactions=test_reactions,
    true_labels=true_labels,
    ec_classes=ec_numbers,
    top_k=100
)

for ec_class, metrics in ec_results.items():
    print(f"{ec_class}: BEDROC_85 = {metrics.bedroc_85:.4f}")
```

### Custom Metrics

```python
from evaluation import EvaluationMetrics

# Create custom metrics
metrics = EvaluationMetrics()
metrics.bedroc_85 = 0.45
metrics.top1_accuracy = 0.25
metrics.auroc = 0.85

# Convert to dictionary
metrics_dict = metrics.to_dict()

# Pretty print
print(metrics)
```

## 📖 API Reference

### EvaluationMetrics

Container for all evaluation metrics.

```python
@dataclass
class EvaluationMetrics:
    bedroc_85: float      # Primary metric
    bedroc_50: float
    bedroc_20: float
    top1_accuracy: float
    top5_accuracy: float
    top10_accuracy: float
    top50_accuracy: float
    top100_accuracy: float
    ef_1pct: float
    ef_5pct: float
    ef_10pct: float
    auroc: float
    auprc: float
    hit_rate_10: float
    hit_rate_50: float
    hit_rate_100: float
    num_actives: int
    num_total: int
    active_fraction: float
```

### compute_all_metrics()

Main function for comprehensive evaluation.

```python
def compute_all_metrics(
    ranked_ids: List[str],
    scores: Union[np.ndarray, torch.Tensor, List[float]],
    active_ids: List[str],
    alpha_values: List[float] = [85.0, 50.0, 20.0],
    top_k_values: List[int] = [1, 5, 10, 50, 100],
    ef_fractions: List[float] = [0.01, 0.05, 0.10]
) -> EvaluationMetrics
```

## 🎓 Best Practices

1. **Use BEDROC₈₅ as primary metric** (matches CLIPZyme paper)
2. **Report multiple metrics** (BEDROC, Top-K, EF, AUROC)
3. **Include confidence intervals** (bootstrap with n≥1000)
4. **Compare to paper results** for validation
5. **Visualize with ROC and PR curves**
6. **Test statistical significance** when comparing models

## 📚 References

- **CLIPZyme Paper**: Mikhael et al. (2024) "CLIPZyme: Reaction-Conditioned Virtual Screening of Enzymes"
- **BEDROC**: Truchon & Bayly (2007) "Evaluating Virtual Screening Methods"

## 💡 Tips

- **BEDROC₈₅ > 0.4**: Good performance (matches paper baseline)
- **BEDROC₈₅ > 0.7**: Excellent performance (matches paper with EC2)
- **Use bootstrap**: n=1000 iterations for stable CI estimates
- **Multiple test correction**: Use Bonferroni for multiple comparisons
- **Stratify analysis**: By EC class, reaction type, etc.

---

For examples, see `scripts/demo_evaluation.py`
