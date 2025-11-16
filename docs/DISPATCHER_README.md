# CLIPZyme Dispatcher System

A complete workflow orchestration system for CLIPZyme virtual screening with advanced resource management, job scheduling, and monitoring.

## Features

- **Task Abstraction**: Define reusable tasks with dependencies
- **Workflow Orchestration**: DAG-based execution with automatic dependency resolution
- **Resource Management**: Intelligent GPU allocation, memory management, and caching
- **Job Scheduling**: Priority-based queue with multiple scheduling strategies
- **Monitoring**: Real-time progress tracking, metrics collection, and structured logging
- **Python API & CLI**: Flexible interfaces for different use cases

## Installation

The dispatcher system is included with the CLIPZyme project. Install dependencies:

```bash
pip install -r requirements.txt
# Additional dependencies for dispatcher
pip install click psutil
```

## Quick Start

### Python API

```python
from dispatcher import DispatcherAPI, create_screening_workflow

# Create dispatcher
dispatcher = DispatcherAPI()

# Create screening workflow
workflow = create_screening_workflow(
    checkpoint_name="clipzyme_official_v1",
    reactions=["CC(=O)O>>CCO", "CC(C)CC(N)C(=O)O>>CC(C)CC(=O)C(=O)O"],
    proteins_csv="data/proteins.csv",
    top_k=100
)

# Submit job
job_id = dispatcher.submit_workflow(workflow, name="my_screening")

# Wait for completion
result = dispatcher.wait_for_job(job_id)

print(f"Status: {result.status}")
print(f"Duration: {result.duration:.2f}s")
```

### Command-Line Interface

```bash
# Start dispatcher
python -m dispatcher.api.cli start

# Run screening
python -m dispatcher.api.cli screen clipzyme_v1 reactions.txt \
    --proteins-csv data/proteins.csv \
    --top-k 100

# Check status
python -m dispatcher.api.cli list-jobs

# Show statistics
python -m dispatcher.api.cli stats
```

## Architecture

```
dispatcher/
├── core/              # Core components
│   ├── task.py       # Task abstraction
│   ├── registry.py   # Task registry
│   ├── workflow.py   # Workflow & DAG
│   ├── executor.py   # Execution engine
│   └── state.py      # State management
├── tasks/            # CLIPZyme-specific tasks
│   ├── checkpoint.py # Checkpoint tasks
│   ├── screening.py  # Screening tasks
│   └── evaluation.py # Evaluation tasks
├── scheduler/        # Job scheduling
│   ├── queue.py      # Job queue
│   ├── scheduler.py  # Scheduler
│   └── priority.py   # Priority strategies
├── resources/        # Resource management
│   ├── gpu.py        # GPU manager
│   ├── memory.py     # Memory manager
│   └── cache.py      # Cache manager
├── monitoring/       # Monitoring & logging
│   ├── logger.py     # Structured logging
│   ├── metrics.py    # Metrics collection
│   ├── progress.py   # Progress tracking
│   └── reporter.py   # Status reporting
├── config/           # Configuration
│   ├── resolver.py   # Config resolution
│   └── validator.py  # Config validation
├── api/              # APIs
│   ├── python_api.py # Python API
│   └── cli.py        # CLI
└── workflows/        # Pre-built workflows
    ├── screening.py  # Screening workflows
    └── evaluation.py # Evaluation workflows
```

## Workflows

### Screening Pipeline

Complete pipeline from checkpoint to results:

```python
from dispatcher.workflows import create_screening_workflow

workflow = create_screening_workflow(
    checkpoint_name="clipzyme_official_v1",
    reactions=["CC(=O)O>>CCO"],
    proteins_csv="data/proteins.csv",
    top_k=100,
    device="cuda",
    mode="interactive"
)
```

**Steps:**
1. Download checkpoint from Zenodo
2. Validate checkpoint integrity
3. Load checkpoint into model
4. Build protein screening set
5. Run virtual screening

### Full Pipeline

Screening + evaluation:

```python
from dispatcher.workflows import create_full_pipeline

workflow = create_full_pipeline(
    checkpoint_name="clipzyme_official_v1",
    reactions=reaction_list,
    proteins_csv="data/proteins.csv",
    ground_truth_csv="data/ground_truth.csv",
    top_k=100
)
```

**Steps:**
1-5. (Same as screening pipeline)
6. Evaluate results against ground truth
7. Generate report with visualizations

## Custom Workflows

Build custom workflows with the `WorkflowBuilder`:

```python
from dispatcher import WorkflowBuilder, TaskConfig
from dispatcher.tasks import LoadCheckpointTask, RunScreeningTask

# Create builder
builder = WorkflowBuilder(
    name="custom_workflow",
    description="My custom screening workflow"
)

# Add tasks
load_task = LoadCheckpointTask(
    config=TaskConfig(
        name="load_checkpoint",
        gpu_required=True
    ),
    checkpoint_path="checkpoints/my_model.pt"
)
builder.add_task(load_task)

screen_task = RunScreeningTask(
    config=TaskConfig(
        name="run_screening",
        depends_on=["load_checkpoint"]
    ),
    reactions=["..."],
    top_k=50
)
builder.add_task(screen_task)

# Build workflow
workflow = builder.build()
```

## Configuration

### Configuration File

Create a YAML configuration file:

```yaml
# dispatcher_config.yaml
scheduler:
  max_concurrent_jobs: 2
  max_workers_per_job: 4

resources:
  gpu:
    auto_allocate: true
    min_free_memory_mb: 1000
  memory:
    reserve_ratio: 0.2
  cache:
    l1_size_mb: 512
    l2_size_mb: 10240

tasks:
  screening:
    mode: interactive
    batch_size: 8
    top_k: 100

logging:
  level: INFO
  console: true
  file: true
```

Use configuration:

```python
dispatcher = DispatcherAPI(config_file="dispatcher_config.yaml")
```

### Runtime Configuration

Override configuration at runtime:

```python
config = {
    'scheduler': {
        'max_concurrent_jobs': 4
    }
}

dispatcher = DispatcherAPI(config=config)
```

## Resource Management

### GPU Allocation

The dispatcher automatically manages GPU allocation:

```python
# Specify GPU requirements in task config
config = TaskConfig(
    name="my_task",
    gpu_required=True,
    min_gpus=1,
    max_gpus=2
)
```

### Memory Management

Automatic memory tracking and batch size suggestions:

```python
from dispatcher.resources import get_memory_manager

mem_manager = get_memory_manager()

# Get suggested batch size
batch_size = mem_manager.suggest_batch_size(
    item_memory_mb=100,
    max_batch_size=64
)
```

### Caching

Two-level cache (memory + disk):

```python
from dispatcher.resources import TwoLevelCache

cache = TwoLevelCache(
    memory_size_mb=512,
    disk_size_mb=10240
)

# Cache embeddings
cache.set("protein_embeddings", embeddings)

# Retrieve
embeddings = cache.get("protein_embeddings")
```

## Monitoring

### Progress Tracking

```python
def progress_callback(percentage, message):
    print(f"{percentage:.1f}% - {message}")

# Use in tasks
task = RunScreeningTask(
    config=config,
    reactions=reactions,
    progress_callback=progress_callback
)
```

### Metrics Collection

```python
from dispatcher.monitoring import get_metrics_collector

metrics = get_metrics_collector()

# Record metrics
metrics.record("screening_time", 45.2, unit="seconds")
metrics.increment("reactions_screened")

# Get statistics
stats = metrics.get_stats("screening_time")
print(f"Average: {stats['mean']:.2f}s")
```

### Structured Logging

```python
from dispatcher.monitoring import configure_logging

configure_logging(
    level=logging.INFO,
    structured=True  # JSON logging
)
```

## Job Scheduling

### Priority Strategies

Available priority strategies:
- `fifo`: First-in-first-out
- `priority`: User-defined priority
- `sjf`: Shortest job first
- `weighted`: Weighted combination
- `fair_share`: Fair distribution across users

### Job Priorities

```python
from dispatcher import TaskPriority

# Submit high-priority job
job_id = dispatcher.submit_workflow(
    workflow,
    priority=TaskPriority.HIGH
)
```

## Examples

See example scripts:
- `scripts/demo_dispatcher_simple.py` - Basic usage
- `scripts/demo_dispatcher_advanced.py` - Advanced features

Run examples:

```bash
python scripts/demo_dispatcher_simple.py
python scripts/demo_dispatcher_advanced.py
```

## API Reference

### DispatcherAPI

Main API class for dispatcher interaction.

**Methods:**
- `submit_workflow(workflow, name, priority)` - Submit workflow
- `submit_task(task, name, priority)` - Submit single task
- `get_job_status(job_id)` - Get job status
- `get_job_result(job_id)` - Get job result
- `wait_for_job(job_id, timeout)` - Wait for completion
- `cancel_job(job_id)` - Cancel job
- `list_jobs(status)` - List jobs
- `get_stats()` - Get statistics

### WorkflowBuilder

Fluent API for building workflows.

**Methods:**
- `add_task(task, depends_on)` - Add task
- `set_max_parallel(count)` - Set parallelism
- `set_fail_fast(value)` - Set fail-fast behavior
- `on_task_complete(callback)` - Set callback
- `build()` - Build workflow

## Best Practices

1. **Use pre-built workflows** when possible
2. **Configure resource limits** appropriate for your hardware
3. **Enable caching** for repeated screenings
4. **Monitor resource usage** to optimize performance
5. **Use structured logging** in production
6. **Set appropriate priorities** for jobs
7. **Handle errors gracefully** with retry logic

## Troubleshooting

### Out of Memory

Reduce batch size in configuration:

```yaml
tasks:
  screening:
    batch_size: 4  # Reduce from default 8
```

### GPU Not Detected

Check CUDA availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### Jobs Stuck in Queue

Increase concurrent jobs:

```yaml
scheduler:
  max_concurrent_jobs: 4  # Increase from default 2
```

## License

Part of the CLIPZyme project. See main project LICENSE.

## Support

For issues and questions, please use the main project issue tracker.
