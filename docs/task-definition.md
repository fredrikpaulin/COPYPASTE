# Task Definition Reference

Tasks are JSON files stored in `tasks/`. Each file defines what kind of model you're training, how to generate synthetic data, and how to configure the training run.

The schema lives at `schemas/task.schema.json` and is validated at load time by `lib/task.js`.

## Required fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Unique identifier. Lowercase letters, digits, hyphens, underscores only. Pattern: `^[a-z0-9_-]+$` |
| `type` | string | One of: `classification`, `extraction`, `regression`, `sequence-labeling` |
| `description` | string | Human-readable description of what the model should do |

## Conditional fields

These are required depending on the task `type`:

| Field | Required when | Type | Description |
|---|---|---|---|
| `labels` | `classification`, `sequence-labeling` | string[] | At least 2 class labels |
| `fields` | `extraction` | string[] | Names of fields to extract from text |

## Optional sections

### `synthetic`

Controls how Claude generates training examples.

| Field | Type | Default | Description |
|---|---|---|---|
| `count` | integer | `100` | Total examples to generate |
| `prompt` | string | _(required)_ | Prompt template. Use `{label}` or `{field}` as placeholders |
| `model` | string | `claude-sonnet-4-20250514` | Anthropic model ID |
| `batchSize` | integer | `10` | Examples per API call |

The `prompt` field supports two placeholders:

- `{label}` — replaced with the list of labels joined by "or" (for classification tasks)
- `{field}` — replaced with the list of fields joined by commas (for extraction tasks)

### `realData`

Points to an existing labeled dataset to mix with synthetic data.

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | string | _(required)_ | Path to a JSONL file |
| `inputField` | string | `text` | Which JSON key holds the input text |
| `labelField` | string | `label` | Which JSON key holds the label |

The real data loader normalizes your field names to the standard `text`/`label` format used internally.

### `training`

Configures the training step.

| Field | Type | Default | Description |
|---|---|---|---|
| `script` | string | `scripts/train.py` | Path to the Python training script |
| `splitRatio` | number | `0.8` | Fraction of data used for training (rest goes to validation). Range: 0.1–0.9 |
| `args` | object | `{}` | Extra key-value pairs passed as CLI flags to the training script |

Extra args are passed as `--key value` pairs. For example, `{"epochs": 5, "lr": 0.01}` becomes `--epochs 5 --lr 0.01`.

## Full example

```json
{
  "name": "intent-classifier",
  "type": "classification",
  "labels": ["greeting", "farewell", "question", "command", "other"],
  "description": "Classify user message intent in a chatbot",
  "synthetic": {
    "count": 500,
    "prompt": "Generate a realistic chatbot user message with intent {label}",
    "model": "claude-sonnet-4-20250514",
    "batchSize": 10
  },
  "realData": {
    "path": "data/real_intents.jsonl",
    "inputField": "message",
    "labelField": "intent"
  },
  "training": {
    "splitRatio": 0.8,
    "args": {
      "max-features": 5000
    }
  }
}
```

## Validation

Tasks are validated against the JSON Schema when loaded (`loadTask`) or saved (`saveTask`). The validator checks:

- Required fields are present
- Types match (string, integer, number, array)
- Enums are respected (e.g. `type` must be one of the four options)
- Name matches the allowed pattern
- Conditional requirements (e.g. `labels` is required when `type` is `classification`)
- Array minimums (e.g. `labels` needs at least 2 items)
- Numeric ranges (e.g. `splitRatio` must be between 0.1 and 0.9)

Defaults are applied after validation for any optional fields you leave out.
