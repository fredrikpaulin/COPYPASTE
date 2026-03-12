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

Once you open a task, the task menu offers four options:

1. **Run full pipeline** — executes steps 2-4 in sequence
2. **Generate synthetic data** — calls Claude to produce labeled JSONL
3. **Prepare data** — merges synthetic + real data, shows stats, splits train/val
4. **Train model** — spawns `scripts/train.py`, outputs a model to `models/<task-name>/`

You can run each step independently. For example, generate data once, then re-train multiple times with different parameters.

## Output

After a successful run, you'll find:

- `data/<name>_synthetic.jsonl` — raw generated examples
- `data/<name>_train.jsonl` — training split
- `data/<name>_val.jsonl` — validation split
- `models/<name>/model.pkl` — serialized scikit-learn pipeline
- `models/<name>/meta.json` — accuracy, label list, dataset sizes
