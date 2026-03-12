# TUI Reference

The terminal interface is built with raw ANSI escape codes in `lib/tui.js`. No dependencies — just `process.stdout.write` and `process.stdin`.

## Controls

| Key | Action |
|---|---|
| `↑` / `↓` | Navigate menu items |
| `Enter` | Select highlighted item |
| `Ctrl-C` | Exit immediately |

Text prompts (task name, labels, etc.) use normal line input — type your answer and press Enter.

## Screens

### Main menu

The entry point. Lists available actions and any existing tasks:

```
╔══════════════════════════════════════╗
║     Feature Distillation Pipeline     ║
╚══════════════════════════════════════╝

  Main Menu
  ❯ Create new task
    Open task: sentiment
    Open task: spam-detector
    Quit
```

### Create new task

A guided flow that prompts for each field in sequence: name, type (arrow-key menu), description, labels/fields (depending on type), generation prompt, example count, and optional real data path.

### Task menu

After opening a task, you see its action menu:

```
  Task: sentiment
  ❯ Run full pipeline
    Generate synthetic data
    Prepare data (merge + split)
    Train model
    ← Back
```

### Generation progress

During synthetic data generation, a progress bar updates after each batch:

```
  ── Generating synthetic data for "sentiment" ──

  → Model: claude-sonnet-4-20250514
  → Target: 50 examples in batches of 10
  ██████████████████░░░░░░░░░░░░░░  60% examples
```

### Data preparation

After merging and before splitting, a statistics table appears:

```
  ── Preparing data for "sentiment" ──

  ──────────────────┼───────
   Metric           │ Count
  ──────────────────┼───────
   Total            │ 250
   Synthetic        │ 200
   Real             │ 50
     label: positive│ 85
     label: negative│ 82
     label: neutral │ 83
  ──────────────────┼───────

  ✓ Train: 200 → data/sentiment_train.jsonl
  ✓ Val: 50 → data/sentiment_val.jsonl
```

### Training

A spinner runs while Python trains the model, replaced by the accuracy readout on completion:

```
  ── Training model for "sentiment" ──

  ✓ Validation Accuracy: 0.9200
  ✓ Model saved to models/sentiment
```

## Status indicators

| Symbol | Meaning |
|---|---|
| `✓` (green) | Success |
| `⚠` (yellow) | Warning |
| `✗` (red) | Error |
| `→` (cyan) | Info |
| `❯` (green) | Currently selected menu item |
| `⠋⠙⠹...` (cyan) | Spinner — operation in progress |

## Colors

The TUI uses a consistent color scheme via ANSI codes:

- **Cyan** — headings, info messages, prompts, spinner frames
- **Green** — success messages, selected items, progress bar fill
- **Yellow** — warnings
- **Red** — errors
- **Magenta** — section headers
- **Dim** — unselected menu items, secondary text, stderr output
