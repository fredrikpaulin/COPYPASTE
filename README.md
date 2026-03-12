# Feature Distillation Pipeline

A zero-dependency TUI application for distilling large language model capabilities into small, specialized models. Define a task, generate synthetic training data via LLM APIs (Claude, OpenAI, or Ollama), optionally mix in real labeled data, then train a lightweight model — all from an interactive terminal interface.

## How it works

```
Task Definition ──→ Synthetic Data Generation ──→ Data Preparation ──→ Training ──→ Evaluation
   (JSON)              (LLM API)               (merge + split)     (Python/sklearn)  (HTML report)
```

You describe _what_ you want your small model to do in a JSON task file — or start from one of the built-in templates (sentiment, intent, spam, contact extraction, topic). The pipeline calls your chosen LLM provider to produce labeled examples matching your specification, merges them with any real data you have, splits into train/validation sets, and spawns a Python training script that outputs a serialized model. Generate an HTML evaluation report to inspect confusion matrices, per-label metrics, and error examples.

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
- At least one LLM provider configured: Anthropic API key (`ANTHROPIC_API_KEY`), OpenAI API key (`OPENAI_API_KEY`), or a running Ollama instance

## Project layout

```
index.js                Entry point — wires TUI to pipeline
lib/
  tui.js                ANSI terminal UI (menus, spinners, tables)
  task.js               Task loading, validation, defaults
  provider.js           Multi-provider LLM abstraction (Claude, OpenAI, Ollama)
  generate.js           LLM data generation + retry/backoff
  data.js               JSONL I/O, merge, split, dedup, augment, semantic dedup
  train.js              Python subprocess + model versioning
  infer.js              Inference via Python subprocess
  bundle.js             Standalone model packaging
  active.js             Active learning / uncertainty sampling
  embed.js              Multi-provider embedding abstraction (OpenAI, Ollama)
  embed-cache.js        SQLite embedding cache (bun:sqlite)
  curriculum.js         Curriculum learning, LLM-as-judge, contrastive generation
  multitask.js          Shared features, zero-shot eval, progressive distillation
  evaluate.js           K-fold CV, feature importance, error taxonomy, calibration
  templates.js          Pre-built task template loading
  report.js             HTML evaluation report generation
  config.js             Project config (distill.config.json)
  log.js                Structured JSONL logging
schemas/
  task.schema.json      JSON Schema for task definitions
templates/              Pre-built task templates (*.json)
tasks/                  Your task definition files (*.json)
data/                   Generated and prepared datasets (*.jsonl)
scripts/
  train.py              scikit-learn training + ONNX export + inference
models/                 Trained model output (created at runtime)
reports/                HTML evaluation reports (created at runtime)
logs/                   Structured run logs (created at runtime)
```
