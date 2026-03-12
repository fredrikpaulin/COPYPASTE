# Architecture

## Design principles

- **Zero JS dependencies** — everything is built with Bun builtins, `fetch`, Node.js standard modules, and raw ANSI escape codes.
- **Minimal abstraction** — each module is a single file with a small, flat API.
- **JSON Schema driven** — task definitions are validated against a schema; the schema is the source of truth for structure and defaults.
- **Python for ML only** — JavaScript handles orchestration, I/O, and UI. Python handles model training. The boundary is a subprocess with JSONL files as the data contract.

## Module map

```
index.js
  ├── lib/tui.js        Terminal rendering (pure side effects)
  ├── lib/task.js        Task CRUD + validation (pure logic + file I/O)
  ├── lib/generate.js    Claude API client (network I/O)
  ├── lib/data.js        Data loading/merging/splitting (file I/O)
  └── lib/train.js       Python subprocess manager (process I/O)
```

No module imports another `lib/` module. They all export functions consumed by `index.js`, which acts as the composition root.

## Data flow

```
                    ┌─────────────────┐
                    │  tasks/*.json   │  Task definitions
                    └────────┬────────┘
                             │ loadTask()
                             ▼
                    ┌─────────────────┐
                    │   generate()    │  Claude API
                    └────────┬────────┘
                             │ writes JSONL
                             ▼
                    ┌─────────────────┐
                    │  data/          │  *_synthetic.jsonl
                    │  (+ real data)  │
                    └────────┬────────┘
                             │ loadAndMerge() + split()
                             ▼
                    ┌─────────────────┐
                    │  data/          │  *_train.jsonl, *_val.jsonl
                    └────────┬────────┘
                             │ runTraining()
                             ▼
                    ┌─────────────────┐
                    │  models/        │  model.pkl, meta.json
                    └─────────────────┘
```

## File conventions

- Task definitions: `tasks/<name>.json`
- Synthetic data: `data/<name>_synthetic.jsonl`
- Train split: `data/<name>_train.jsonl`
- Validation split: `data/<name>_val.jsonl`
- Model artifacts: `models/<name>/model.pkl` + `models/<name>/meta.json`

The `<name>` is always the task's `name` field, which is constrained to `[a-z0-9_-]+`.

## lib/tui.js

Renders everything using ANSI escape codes written directly to `process.stdout`. Key exports:

- `menu(title, items)` — arrow-key navigable menu, returns selected index
- `prompt(question)` — single-line text input
- `progress(current, total, label)` — overwriting progress bar
- `spinner(label)` — animated spinner with `.stop()` and `.fail()`
- `table(rows, headers)` — bordered ASCII table
- `banner()`, `header()`, `success()`, `warn()`, `error()`, `info()`, `dim()` — styled output

The TUI uses raw mode for menus (to capture arrow keys) and cooked mode for text prompts.

## lib/task.js

Loads JSON task files, validates them against the schema, and applies defaults. The validator is a minimal JSON Schema subset implementation — just enough to cover the features used in `task.schema.json` (types, enums, patterns, required fields, conditional requirements via `allOf`/`if`/`then`, and numeric ranges).

Exports: `loadTask(name)`, `listTasks()`, `saveTask(task)`, `validate(data, schema)`, `loadSchema()`.

## lib/generate.js

Calls the Anthropic Messages API via `fetch`. Builds task-appropriate system and user prompts, sends sequential batch requests, parses JSON arrays from responses, and writes accumulated results to JSONL.

Exports: `generate(task, { apiKey, onProgress })`.

## lib/data.js

Reads and writes JSONL files using `Bun.file`. Handles merging synthetic and real data (with field normalization), Fisher-Yates shuffling, train/val splitting, and computing label distribution stats.

Exports: `loadAndMerge(task)`, `split(task, data)`, `stats(data)`, `readJsonl(path)`, `writeJsonl(path, rows)`.

## lib/train.js

Spawns `python3` as a child process with CLI arguments built from the task definition. Streams stdout/stderr via callbacks so the TUI can display progress. Returns a promise that resolves with the model directory path on success.

Exports: `runTraining(task, trainPath, valPath, { onStdout, onStderr })`.

## scripts/train.py

A standalone Python script that reads JSONL, trains a model, evaluates on validation data, and serializes the result. Uses `argparse` for CLI, `pickle` for model serialization, and scikit-learn for the ML pipeline. Designed to be replaceable — any script that accepts the same flags will work.
