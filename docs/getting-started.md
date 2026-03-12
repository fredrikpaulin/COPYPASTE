# Getting Started

## Prerequisites

**Bun** — the JavaScript runtime. Install from [bun.sh](https://bun.sh) or via npm:

```bash
npm install -g bun
```

**Python 3.8+** with scikit-learn — used only for the training step:

```bash
pip install scikit-learn
```

**Anthropic API key** — needed for synthetic data generation. Get one at [console.anthropic.com](https://console.anthropic.com).

## Setup

Clone or copy the project, then set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

No `bun install` needed — the project has zero JavaScript dependencies.

## Configuration (optional)

Create a `distill.config.json` at the project root to set defaults:

```json
{
  "model": "claude-sonnet-4-20250514",
  "batchSize": 10,
  "maxRetries": 3,
  "splitRatio": 0.8,
  "dataDir": "data",
  "modelsDir": "models"
}
```

All fields are optional — anything you leave out uses the built-in defaults shown above.

## Running the TUI

```bash
bun index.js
```

You'll see the main menu:

```
╔══════════════════════════════════════╗
║     Feature Distillation Pipeline     ║
╚══════════════════════════════════════╝

  Main Menu
  ❯ Create new task
    Open task: sentiment
    Quit
```

Use **arrow keys** to move, **Enter** to select, **Ctrl-C** to exit at any time.

## Your first task

### Option A: Use the included example

The project ships with `tasks/sentiment.json` — a sentiment classifier. Open it from the main menu, then select **Run full pipeline**. This will generate 50 synthetic reviews via Claude, split them 80/20, and train a TF-IDF + logistic regression model.

### Option B: Create a new task interactively

Select **Create new task** from the main menu. The TUI walks you through each field: name, type, labels, generation prompt, and example count. Your task is saved to `tasks/<name>.json` and immediately available.

### Option C: Write the JSON by hand

Create a file in `tasks/`:

```json
{
  "name": "spam-detector",
  "type": "classification",
  "labels": ["spam", "not-spam"],
  "description": "Classify whether an email is spam",
  "synthetic": {
    "count": 200,
    "prompt": "Generate a realistic email that is {label}",
    "batchSize": 10
  }
}
```

See [Task Definition](./task-definition.md) for the full schema reference.

## Pipeline steps

Once you open a task, the task menu offers these options:

1. **Run full pipeline** — executes generate → prepare → train in sequence
2. **Preview data** — generate a small sample to inspect before a full run
3. **Generate synthetic data** — calls Claude to produce labeled JSONL
4. **Prepare data** — merges synthetic + real data, deduplicates, shows stats, checks label balance, splits train/val
5. **Augment data** — expand the dataset via synonym replacement and random insertion
6. **Confidence filter** — score examples via Claude and remove low-quality ones
7. **Train model** — spawns `scripts/train.py`, outputs a model to `models/<task-name>/`
8. **Predict** — interactive REPL to test the trained model
9. **Model versions** — list, inspect, or rollback model snapshots
10. **Bundle for deployment** — package the model as a standalone module

You can run each step independently. For example, generate data once, augment and filter, then re-train multiple times.

## Output

After a successful run, you'll find:

- `data/<name>_synthetic.jsonl` — raw generated examples
- `data/<name>_train.jsonl` — training split
- `data/<name>_val.jsonl` — validation split
- `models/<name>/model.pkl` — serialized scikit-learn pipeline
- `models/<name>/meta.json` — accuracy, label list, dataset sizes, timestamp
- `models/<name>/model.onnx` — ONNX export (if enabled)
- `models/<name>/versions/` — timestamped model snapshots
- `logs/<name>_<timestamp>.jsonl` — structured run log
