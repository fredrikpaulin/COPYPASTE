# Changelog

## 0.1.2

### Multi-provider generation (Phase 6)

- Provider abstraction (`lib/provider.js`) — unified `callProvider()` interface with per-provider fetch implementations
- Anthropic (Claude), OpenAI, and Ollama support out of the box
- Provider auto-detection from task config (`synthetic.provider` field) or project config
- `listProviders()` shows available providers and their configuration status
- `resolveProvider()` merges task config, project config, and environment variables

### Task templates (Phase 6)

- Five pre-built task templates: sentiment, intent, spam-detector, contact-extractor, topic
- `listTemplates()` and `loadTemplate()` in `lib/templates.js`
- "Create from template" option in the main TUI menu — clone and customize a template to get started fast

### Evaluation dashboard (Phase 6)

- `lib/report.js` — generates standalone HTML evaluation reports
- Confusion matrix with color-coded cells
- Per-label precision, recall, F1, and support metrics
- Label distribution bar chart
- Example errors grouped by misclassification type
- "Evaluation report" option in the task menu

### Tests

- New `test/phase6.test.js` — provider (10 tests with mock HTTP servers), templates (4 tests), report (4 tests)
- 162 tests passing across 10 files

## 0.1.1

### Model capabilities (Phase 3)

- Multi-algorithm support — train with logistic regression, SVM (LinearSVC), or random forest
- Model comparison — train all three algorithms and auto-select the best by validation accuracy
- Hyperparameter grid search — sweep over parameter combinations with results table
- Extraction task training — per-field binary classifiers using TF-IDF + logistic regression
- Prediction handles extraction models (dict of per-field pipelines) and SVM (no predict_proba)

### Active learning (Phase 5)

- Uncertainty sampling — generate candidates, rank by model confidence, surface the most uncertain
- LLM-in-the-loop labeling — send uncertain examples to Claude for corrected labels
- Active learning history — track iterations (examples added, accuracy before, method, timestamp)
- TUI screens for uncertainty sampling and iteration history

### Production hardening (Phase 1)

- Exponential backoff with jitter for Claude API retries (429, 529, 5xx)
- Respect `retry-after` headers from the API
- Input validation — malformed generated examples are dropped and reported instead of breaking the batch
- Structured JSONL logging to `logs/` for every pipeline run
- Project-level `distill.config.json` for default model, batch size, retries, directories
- Updated `.gitignore` for data, models, and config

### Data quality (Phase 2)

- `preview` command — generate a small sample and inspect before committing to a full run
- Exact and fuzzy deduplication (trigram Jaccard similarity, configurable threshold)
- Data augmentation — synonym replacement and random word insertion with configurable multiplier
- Label imbalance detection with warnings in the TUI
- Confidence filtering — score examples via Claude and drop low-confidence rows

### Deployment (Phase 4)

- ONNX export via `--onnx` flag in the training script
- Inference module (`lib/infer.js`) with interactive predict REPL in the TUI
- Model versioning with timestamped snapshots and rollback
- Bundled deployment — package model + standalone `predict.py` as a drop-in module

### Tests

- Rewrote `test/generate.test.js` — added validateExample, backoffMs, retry, and dropped-example tests
- New `test/data-quality.test.js` — trigrams, deduplication, augmentation, label balance
- New `test/config-log.test.js` — config loading, caching, defaults, structured logging
- New `test/phase3.test.js` — algorithm selection, model comparison, hyperparameter search, extraction training
- New `test/active.test.js` — uncertainty sampling, iteration history tracking
- 143 tests passing across 9 files

## 0.1.0

Initial release — foundation.

- Task definition via JSON Schema (classification, extraction, regression, sequence-labeling)
- Synthetic data generation via Claude Messages API (direct fetch, no SDK)
- JSONL data pipeline with real-data loading, normalization, merge, and train/val split
- scikit-learn training (TF-IDF + LogisticRegression) via Python subprocess
- Interactive TUI built from raw ANSI escape codes (menus, spinners, progress bars, tables)
- `/docs` documentation suite
- `bun test` suite (86 tests)
