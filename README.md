# Feature Distillation Pipeline

A zero-dependency TUI application for distilling large language model capabilities into small, specialized models. Define a task, generate synthetic training data via the Claude API, optionally mix in real labeled data, then train a lightweight model — all from an interactive terminal interface.

## How it works

```
Task Definition ──→ Synthetic Data Generation ──→ Data Preparation ──→ Training
   (JSON)              (Claude API)              (merge + split)     (Python/sklearn)
```

You describe _what_ you want your small model to do in a JSON task file. The pipeline calls Claude to produce labeled examples matching your specification, merges them with any real data you have, splits into train/validation sets, and spawns a Python training script that outputs a serialized model.

## Documentation

- [Getting Started](./docs/getting-started.md) — install, configure, run your first task
- [Task Definition](./docs/task-definition.md) — schema reference for task JSON files
- [Synthetic Data Generation](./docs/synthetic-data.md) — how Claude produces training examples
- [Data Pipeline](./docs/data-pipeline.md) — loading, merging, splitting, and the JSONL format
- [Training](./docs/training.md) — the Python training script and how to customize it
- [Architecture](./docs/architecture.md) — project structure, module responsibilities, data flow
- [TUI Reference](./docs/tui-reference.md) — keyboard controls and screen descriptions

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-ant-...
bun index.js
```

Use arrow keys to navigate, Enter to select, Ctrl-C to quit.

## Requirements

- [Bun](https://bun.sh) (tested with v1.3+)
- Python 3.8+ with scikit-learn (`pip install scikit-learn`)
- An Anthropic API key for synthetic data generation

## Project layout

```
index.js                Entry point — wires TUI to pipeline
lib/
  tui.js                ANSI terminal UI (menus, spinners, tables)
  task.js               Task loading, validation, defaults
  generate.js           Claude API calls for synthetic data
  data.js               JSONL read/write, merge, shuffle, split
  train.js              Python subprocess orchestration
schemas/
  task.schema.json      JSON Schema for task definitions
tasks/                  Your task definition files (*.json)
data/                   Generated and prepared datasets (*.jsonl)
scripts/
  train.py              scikit-learn training script
models/                 Trained model output (created at runtime)
```
