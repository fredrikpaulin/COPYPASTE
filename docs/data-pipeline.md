# Data Pipeline

The data module (`lib/data.js`) handles loading, normalizing, merging, splitting, and inspecting datasets. Everything is stored in JSONL format.

## JSONL format

All data files use JSON Lines — one JSON object per line. This makes files easy to inspect, append to, and stream.

**Classification example:**
```jsonl
{"text": "Great product!", "label": "positive"}
{"text": "Terrible experience.", "label": "negative"}
```

**Extraction example:**
```jsonl
{"text": "Call me at 555-0123", "fields": {"phone": "555-0123"}}
```

Internally, every row gets a `_source` field (`"synthetic"` or `"real"`) used for tracking provenance in statistics.

## Loading and merging

The `loadAndMerge(task)` function collects data from two possible sources:

1. **Synthetic data** at `data/<task-name>_synthetic.jsonl` — produced by the generation step.
2. **Real data** at the path specified in `task.realData.path` — your own labeled dataset.

Real data is normalized to match the synthetic format. If your real data uses different field names (e.g. `message` instead of `text`, `sentiment` instead of `label`), configure `inputField` and `labelField` in the task's `realData` section:

```json
{
  "realData": {
    "path": "data/my_reviews.jsonl",
    "inputField": "message",
    "labelField": "sentiment"
  }
}
```

Both sources are tagged with `_source` and concatenated into a single array.

## Splitting

The `split(task, data)` function:

1. Shuffles the merged data using Fisher-Yates.
2. Splits at the configured ratio (default 80/20).
3. Writes `data/<task-name>_train.jsonl` and `data/<task-name>_val.jsonl`.

The split ratio is set via `task.training.splitRatio` (range: 0.1–0.9). A value of `0.8` means 80% training, 20% validation.

Each call to `split` re-shuffles and overwrites previous splits.

## Statistics

The `stats(data)` function returns a summary of the merged dataset before splitting:

```js
{
  total: 250,
  bySrc: { synthetic: 200, real: 50 },
  byLabel: { positive: 85, negative: 82, neutral: 83 }
}
```

The TUI renders this as a table so you can verify label distribution and source balance before training.

## File locations

All data files live in the `data/` directory at the project root:

| File | Description |
|---|---|
| `<name>_synthetic.jsonl` | Raw synthetic output from Claude |
| `<name>_train.jsonl` | Training split (shuffled) |
| `<name>_val.jsonl` | Validation split (shuffled) |

Real data files can live anywhere — just reference them by path in the task definition.

## Utility exports

The module also exports `readJsonl(path)` and `writeJsonl(path, rows)` for use in scripts or extensions.
