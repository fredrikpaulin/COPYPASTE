# Training Pipeline

Training is handled by a Python script (`scripts/train.py`) spawned as a subprocess from `lib/train.js`. The JS side manages the lifecycle and streams output to the TUI; the Python side does the actual model fitting.

## How it works

1. `lib/train.js` spawns `python3 scripts/train.py` with CLI arguments derived from the task definition.
2. stdout is streamed back to the TUI (spinner, accuracy readout).
3. stderr is shown as dimmed text for warnings/diagnostics.
4. On success, the model and metadata are written to `models/<task-name>/`.

## CLI arguments

The training script receives these flags:

| Flag | Source | Description |
|---|---|---|
| `--train` | `data/<name>_train.jsonl` | Training data path |
| `--val` | `data/<name>_val.jsonl` | Validation data path |
| `--output` | `models/<name>/` | Output directory for model artifacts |
| `--task-type` | `task.type` | One of: classification, extraction, regression, sequence-labeling |
| `--labels` | `task.labels` | Comma-separated label list (if applicable) |

Any extra args from `task.training.args` are appended as `--key value` pairs.

## Classification training

The default script trains a scikit-learn pipeline with two stages:

1. **TF-IDF vectorizer** — converts text to sparse feature vectors. Configured with `max_features=10000` and bigrams (`ngram_range=(1, 2)`).
2. **Logistic regression** — linear classifier with `max_iter=1000` and `C=1.0` regularization.

After fitting, the script evaluates on the validation set and prints accuracy plus a full classification report (precision, recall, F1 per label).

## Output artifacts

A successful training run produces:

**`models/<name>/model.pkl`** — the serialized scikit-learn Pipeline object. Load it in Python with:

```python
import pickle
with open('models/sentiment/model.pkl', 'rb') as f:
    model = pickle.load(f)

prediction = model.predict(["This product is amazing!"])
```

**`models/<name>/meta.json`** — training metadata:

```json
{
  "task_type": "classification",
  "labels": ["positive", "negative", "neutral"],
  "accuracy": 0.92,
  "train_size": 400,
  "val_size": 100
}
```

## Extraction tasks

Extraction training is currently a placeholder — it saves the train/val data as JSON files in the model directory for use with a custom pipeline. This is a natural extension point.

## Using a custom training script

Set `task.training.script` to point at your own script:

```json
{
  "training": {
    "script": "scripts/my_custom_trainer.py",
    "args": {
      "epochs": 10,
      "learning-rate": 0.001
    }
  }
}
```

Your script will receive the same `--train`, `--val`, `--output`, `--task-type`, and `--labels` flags, plus any extra args you define. Write to stdout for status updates and the TUI will display them.

## Requirements

The only Python dependency is scikit-learn:

```bash
pip install scikit-learn
```

This pulls in numpy, scipy, joblib, and threadpoolctl as transitive dependencies. No deep learning frameworks are needed for the default pipeline.

## Extending to deep learning

To train a neural model instead, replace `scripts/train.py` with your own script that uses PyTorch, TensorFlow, or any other framework. The JS orchestration layer doesn't care what Python does internally — it just needs a process that reads JSONL, trains a model, and writes output to the specified directory.
