# Changelog

## 0.1.8

### Transformer distillation (Phase 12)

- Fine-tune pretrained transformer models (DistilBERT, TinyBERT, BERT-base, RoBERTa, MiniLM) or any HuggingFace model
- GPU support — automatic CUDA/MPS detection with CPU fallback
- Runtime dependency detection — torch and transformers are optional, checked at runtime with clear install instructions
- `scripts/train_transformer.py` — standalone fine-tuning script following the same CLI contract as `train.py`
- `lib/transformer.js` — JS orchestration: dependency checks, device detection, model presets, training, prediction
- HuggingFace Trainer with per-epoch evaluation, structured JSON output for TUI progress (`__EPOCH_START__`, `__TRAIN_LOG__`, `__EVAL_LOG__`, `__TRANSFORMER_RESULTS__`)
- ONNX export via `optimum` (preferred) or `torch.onnx` fallback
- Automatic experiment recording with `algorithm: "transformer:<model>"` and `feature_mode: "transformer"`
- `predictTransformer()` — inference via fine-tuned model with softmax confidence scores
- TUI screens: "Train transformer (fine-tune)", "Predict (transformer)", "Compare all models"
- Task menu expanded to 35 items (indices 0-34)

### Tests

- New `test/phase12.test.js` — exports (2), listModelPresets (3), checkDeps (1), detectDevice (1), hasTransformerModel (1), Python utility modes (4), JS/Python preset consistency (2), error handling (2)
- 262 tests passing across 16 files

## 0.1.7

### Ensemble inference (Phase 11)

- Train all three algorithms (logistic regression, SVM, random forest) into side-by-side ensemble models
- `ensemblePredict()` — weighted majority vote across all trained algorithms, weighted by validation accuracy
- Agreement tracking — what fraction of models agree on the winning label
- Confidence threshold rejection — predictions below a configurable threshold return `rejected: true` instead of a label
- `predictWithThreshold()` — single-model prediction with rejection for low-confidence outputs
- TUI screens: "Train ensemble", "Predict (ensemble)", "Predict (confidence threshold)"

### Experiment tracking (Phase 11)

- SQLite-backed experiment log (`lib/experiment.js`) via `bun:sqlite` — records every training run automatically
- Each experiment stores: task, algorithm, accuracy, train/val size, data fingerprint (FNV-1a hash), feature mode, dimensionality reduction, hyperparams, duration, labels, notes
- `hashDataset()` — fast deterministic fingerprint over all texts and labels in a dataset
- `listExperiments()`, `getExperiment()`, `bestExperiment()`, `experimentStats()` for querying history
- `compareExperiments()` — side-by-side diff showing accuracy delta, data/algorithm changes
- Automatic recording on every `runTrain()` and ensemble training call
- TUI screen: "Experiment history" with comparison workflow

### Tests

- New `test/phase11.test.js` — ensemble exports (5), hashDataset (5), experiment CRUD (10), hyperparams serialization (2)
- 246 tests passing across 15 files

## 0.1.6

### Evaluation and interpretability (Phase 10)

- K-fold cross-validation (`kFoldCV`) — split data into k folds, train/evaluate each, report mean ± std accuracy
- `kFoldSplit()` with Fisher-Yates shuffle and disjoint validation sets
- Feature importance extraction via inline Python — top TF-IDF coefficients per label, random forest importances
- Error taxonomy — categorize misclassifications by confusion pair, text length, and data provider
- Calibration analysis — reliability bins with Expected Calibration Error (ECE)
- 2D data map projection (`projectTo2D`) — power iteration PCA in pure JS for embedding visualization
- TUI screens: "K-fold CV", "Feature importance", "Error taxonomy", "Calibration analysis"

### Tests

- New `test/phase10.test.js` — kFoldSplit (5), errorTaxonomy (6), calibrationBins (4), projectTo2D (4), export checks (2)
- 224 tests passing across 14 files

## 0.1.5

### Multi-task and transfer learning (Phase 9)

- Shared feature training (`sharedFeatureTraining`) — collect texts across tasks for shared TF-IDF vocabulary
- Zero-shot bootstrap (`zeroShotEval`) — evaluate LLM directly on validation data as accuracy baseline
- Progressive distillation (`progressiveDistill`) — chain large provider → local Ollama → tag provenance for staged distillation
- TUI screen: "Zero-shot eval"

### Tests

- New `test/phase9.test.js` — sharedFeatureTraining (2), zeroShotEval structure (2), progressiveDistill export (1)

## 0.1.4

### Curriculum and data strategy (Phase 8)

- Curriculum learning — `scoreDifficulty()` uses model confidence as difficulty proxy, `sortByCurriculum()` orders easy→hard
- `curriculumStages()` splits data into easy/medium/hard buckets with configurable thresholds
- LLM-as-judge quality scoring (`llmJudge`) — rates examples on relevance, naturalness, label correctness rubric via callProvider
- `filterByQuality()` removes examples below a quality threshold
- Contrastive example generation (`generateContrastive`) — hard negatives near decision boundaries between label pairs
- Cross-provider ensembling (`ensembleGenerate`) — generate from multiple providers, tag `_provider` provenance
- TUI screens: "Curriculum analysis", "LLM-as-judge", "Contrastive generation", "Ensemble generate"

### Tests

- New `test/phase8.test.js` — sortByCurriculum (3), curriculumStages (3), filterByQuality (3), generateContrastive validation (2), llmJudge integration (1)

## 0.1.3

### Sentence embeddings (Phase 7)

- Embedding abstraction (`lib/embed.js`) — unified interface for OpenAI and Ollama embedding APIs
- `embed()` function with batching, progress callbacks, and per-provider response parsing
- `resolveEmbedProvider()` and `listEmbedProviders()` for provider configuration
- `cosineSimilarity()` for vector comparison

### Embedding cache (Phase 7)

- SQLite-backed embedding cache (`lib/embed-cache.js`) via `bun:sqlite` — zero dependencies
- Keyed by (text_hash, model) for efficient dedup across re-training runs
- `cachedEmbed()` wrapper that transparently caches embeddings — only embeds texts not already cached
- Batch get/put operations, per-model count and clear
- Float32Array binary packing for compact storage

### Semantic deduplication (Phase 7)

- `semanticDeduplicate()` in `lib/data.js` — cosine similarity on embeddings catches paraphrases that trigram matching misses
- Configurable similarity threshold (default 0.92)
- "Semantic dedup" option in the TUI task menu

### Embedding-based training (Phase 7)

- Training path that uses pre-computed embeddings as features instead of TF-IDF
- `load_embeddings()` and `reduce_dimensions()` in `scripts/train.py`
- Optional PCA or truncated SVD dimensionality reduction with configurable target dimensions
- `--train-embeddings`, `--val-embeddings`, `--dim-reduce`, `--n-components` CLI flags
- "Train model (embeddings)" option in the TUI — embeds texts, writes JSONL, trains classifier

### TUI updates

- Task menu expanded to 19 items: added "Semantic dedup", "Train model (embeddings)", "Embedding cache stats"
- Existing "Train model" renamed to "Train model (TF-IDF)" for clarity

### Tests

- New `test/phase7.test.js` — 24 tests covering cosine similarity (5), provider resolution (3), provider listing (1), mock OpenAI integration (2), mock Ollama integration (1), cache CRUD (7), pack/unpack (1), semantic dedup (2), Python embedding training (2)
- 186 tests passing across 11 files

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
