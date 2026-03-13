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
  ├── lib/provider.js    Multi-provider LLM abstraction (network I/O)
  ├── lib/generate.js    LLM data generation + retry logic (network I/O)
  ├── lib/data.js        Data loading/merging/splitting/dedup/augment (file I/O)
  ├── lib/train.js       Python subprocess + model versioning (process I/O)
  ├── lib/infer.js       Inference via Python subprocess (process I/O)
  ├── lib/bundle.js      Standalone model packaging (file I/O)
  ├── lib/active.js      Active learning / uncertainty sampling (process + network I/O)
  ├── lib/embed.js       Multi-provider embedding abstraction (network I/O)
  ├── lib/embed-cache.js SQLite embedding cache (file I/O)
  ├── lib/curriculum.js  Curriculum learning + data strategy (process + network I/O)
  ├── lib/multitask.js   Multi-task / transfer learning (process + network I/O)
  ├── lib/evaluate.js    Evaluation + interpretability (process I/O + pure logic)
  ├── lib/ensemble.js    Ensemble inference + confidence threshold (process I/O)
  ├── lib/experiment.js  Experiment tracking via SQLite (file I/O)
  ├── lib/transformer.js Transformer fine-tuning orchestration (process I/O)
  ├── lib/templates.js   Pre-built task template loading (file I/O)
  ├── lib/report.js      HTML evaluation report generation (file I/O)
  ├── lib/config.js      Config file loading + defaults (file I/O)
  └── lib/log.js         Structured JSONL logging (file I/O)
```

`lib/generate.js` imports from `lib/provider.js` for multi-provider support. `lib/embed-cache.js` imports from `lib/embed.js` for cached embedding. `lib/data.js` imports from `lib/embed.js` for semantic deduplication. `lib/curriculum.js` imports from `lib/infer.js` (difficulty scoring), `lib/provider.js` (LLM-as-judge, contrastive), and `lib/generate.js` (ensemble). `lib/multitask.js` imports from `lib/provider.js` (zero-shot eval) and `lib/generate.js` (progressive distillation). `lib/evaluate.js` is mostly self-contained — k-fold CV shells out to Python, feature importance runs an inline Python script, and the pure-JS functions (error taxonomy, calibration, PCA projection) have no imports. `lib/transformer.js` orchestrates `scripts/train_transformer.py` via subprocess — handles dependency detection, device discovery, model presets, and structured output parsing. All modules export functions consumed by `index.js`, which acts as the composition root.

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
                    └────────┬────────┘
                             │ predict() / bundle()
                             ▼
                    ┌─────────────────┐
                    │  inference /    │  Interactive predict, bundled deploy
                    │  deployment     │
                    └─────────────────┘
```

## File conventions

- Task definitions: `tasks/<name>.json`
- Synthetic data: `data/<name>_synthetic.jsonl`
- Train split: `data/<name>_train.jsonl`
- Validation split: `data/<name>_val.jsonl`
- Model artifacts: `models/<name>/model.pkl` + `models/<name>/meta.json`
- Model versions: `models/<name>/versions/<timestamp>/`
- Config: `distill.config.json` (project root, optional)
- Logs: `logs/<name>_<timestamp>.jsonl`

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

## lib/provider.js

Multi-provider LLM abstraction supporting Anthropic (Claude), OpenAI, and Ollama. Each provider has its own fetch implementation matching the provider's API format. The unified `callProvider()` function handles retries with exponential backoff and jitter on retryable errors (429, 529, 5xx). `resolveProvider()` merges task config, project config, and environment variables to select the right provider, model, and API key. `listProviders()` enumerates all providers and whether they're configured (have the required API key in env).

Exports: `callProvider(providerName, opts)`, `resolveProvider(task, config)`, `listProviders()`, `PROVIDERS`, `backoffMs(attempt, base, max)`.

## lib/generate.js

Generates synthetic training data via LLM API calls. Uses `lib/provider.js` for multi-provider support — routes to the correct provider based on task config. Builds task-appropriate system and user prompts, sends sequential batch requests, parses JSON arrays from responses, validates each example against the task definition, drops malformed rows, and writes accumulated results to JSONL.

Exports: `generate(task, { apiKey, onProgress, onRetry, onDropped, provider })`, `preview(task, { apiKey, count, onRetry, provider })`, `buildSystemPrompt(task)`, `buildBatchPrompt(task, batchSize)`, `parseBatchResponse(text)`, `validateExample(example, task)`, `backoffMs(attempt, base, max)`.

## lib/data.js

Reads and writes JSONL files using `Bun.file`. Handles merging synthetic and real data (with field normalization), Fisher-Yates shuffling, train/val splitting, computing label distribution stats, exact and fuzzy deduplication (trigram Jaccard similarity), text augmentation (synonym replacement, random word insertion), label imbalance detection, and confidence-based filtering via Claude.

Exports: `loadAndMerge`, `split`, `stats`, `readJsonl`, `writeJsonl`, `deduplicate`, `trigrams`, `trigramSimilarity`, `augment`, `synonymReplace`, `randomInsert`, `labelCounts`, `labelImbalance`, `filterByConfidence`.

## lib/train.js

Spawns `python3` as a child process with CLI arguments built from the task definition. Streams stdout/stderr via callbacks so the TUI can display progress. Supports ONNX export via an `onnx` flag. Also handles model versioning: snapshotting the current model before re-training, listing all versions, and rolling back to a previous version.

Exports: `runTraining(task, trainPath, valPath, { onStdout, onStderr, onnx })`, `versionModel(taskName)`, `listVersions(taskName)`, `rollbackModel(taskName, version)`.

## lib/infer.js

Inference via Python subprocess. Writes texts as JSONL to stdin of `scripts/train.py --predict`, reads JSON predictions from stdout. Also provides `loadMeta(taskName)` to read model metadata and `listModels()` to enumerate trained models.

Exports: `predict(taskName, texts)`, `loadMeta(taskName)`, `listModels()`.

## lib/bundle.js

Packages a trained model as a standalone, drop-in module. Copies model artifacts, generates a self-contained `predict.py` script, a `package.json` with metadata, and a `README.md` with usage instructions.

Exports: `bundle(taskName, outputDir)`.

## lib/active.js

Active learning via uncertainty sampling. Runs predictions through the trained model, ranks by confidence (ascending), and surfaces the most uncertain examples for labeling. Supports LLM-in-the-loop labeling (send uncertain examples to Claude), iteration history tracking, and integration with the training data pipeline.

Exports: `getUncertainExamples(taskName, texts, { topK })`, `generateAndRankByUncertainty(task, opts)`, `llmLabel(examples, task, { apiKey, model })`, `loadHistory(taskName)`, `saveIteration(taskName, iteration)`.

## lib/embed.js

Multi-provider embedding abstraction supporting OpenAI (`text-embedding-3-small`) and Ollama (`nomic-embed-text`). Each provider has its own fetch implementation matching the provider's embedding API format. Includes `cosineSimilarity()` for vector comparison, used by semantic deduplication. Batch embedding with configurable chunk size and progress callbacks.

Exports: `embed(providerName, texts, opts)`, `cosineSimilarity(a, b)`, `resolveEmbedProvider(task, config)`, `listEmbedProviders()`, `EMBEDDING_PROVIDERS`.

## lib/embed-cache.js

SQLite-backed embedding cache using `bun:sqlite`. Stores embeddings as packed Float32Array buffers keyed by (text_hash, model). Avoids re-embedding the same text when re-training or experimenting. `cachedEmbed()` wraps the embed function — looks up cached vectors first, only calls the API for misses, stores new results.

Exports: `createEmbedCache(cachePath)`, `cachedEmbed(providerName, texts, opts)`, `hashText(text)`, `packEmbedding(arr)`, `unpackEmbedding(buf)`.

## lib/curriculum.js

Curriculum learning and data strategy. `scoreDifficulty()` runs a trained model over the dataset and uses `1 - confidence` as a difficulty proxy. `sortByCurriculum()` orders data easy-first for staged training. `curriculumStages()` splits into easy/medium/hard buckets. `llmJudge()` scores examples on a relevance/naturalness/label-correctness rubric via `callProvider`. `filterByQuality()` removes low-scoring examples. `generateContrastive()` produces hard negatives near decision boundaries between label pairs. `ensembleGenerate()` generates data from multiple providers and tags `_provider` provenance.

Exports: `scoreDifficulty(taskName, data)`, `sortByCurriculum(scoredData)`, `curriculumStages(scoredData, opts)`, `llmJudge(data, task, opts)`, `filterByQuality(scoredData, threshold)`, `generateContrastive(task, opts)`, `ensembleGenerate(task, providers, opts)`.

## lib/multitask.js

Multi-task and transfer learning. `sharedFeatureTraining()` collects training texts across multiple tasks for a shared TF-IDF vocabulary. `zeroShotEval()` evaluates the LLM directly on validation data as a baseline — no training needed — to establish how much distillation helps. `progressiveDistill()` chains a large provider → local Ollama model, tagging provenance at each stage for downstream analysis.

Exports: `sharedFeatureTraining(tasks, opts)`, `zeroShotEval(task, valData, opts)`, `progressiveDistill(task, opts)`.

## lib/evaluate.js

Evaluation and interpretability. `kFoldSplit()` creates k disjoint validation folds with Fisher-Yates shuffle. `kFoldCV()` runs training/evaluation for each fold via the Python subprocess and reports mean ± std accuracy. `featureImportance()` runs an inline Python script to extract top TF-IDF coefficients per label (or random forest importances). `errorTaxonomy()` categorizes misclassifications by confusion pair, text length, and data source. `calibrationBins()` computes reliability bins and Expected Calibration Error (ECE). `projectTo2D()` uses power iteration PCA in pure JS to project embeddings onto two principal components for data map visualization.

Exports: `kFoldSplit(n, k)`, `kFoldCV(task, data, opts)`, `featureImportance(taskName, opts)`, `errorTaxonomy(valData, predictions)`, `calibrationBins(predictions, actual, opts)`, `projectTo2D(embeddings)`.

## lib/ensemble.js

Ensemble inference combining predictions from multiple trained algorithms. `trainEnsembleModels()` trains all three algorithms (logistic regression, SVM, random forest) and stores them in `models/<task>/ensemble/<algorithm>/`. `ensemblePredict()` runs all ensemble models and combines via weighted majority vote — weights are each model's validation accuracy. Returns confidence (weighted vote share), agreement ratio (fraction of models agreeing), and optional rejection when confidence falls below a threshold. `predictWithThreshold()` wraps the primary model's prediction with a rejection mechanism for low-confidence outputs.

Exports: `trainEnsembleModels(task, trainPath, valPath, opts)`, `listEnsembleModels(taskName)`, `ensemblePredict(taskName, texts, opts)`, `predictWithThreshold(taskName, texts, opts)`, `ALGORITHMS`.

## lib/experiment.js

SQLite-backed experiment tracking using `bun:sqlite`. Every training run is automatically recorded with task name, algorithm, accuracy, train/val size, data fingerprint, feature mode, dimensionality reduction settings, hyperparameters (JSON), duration, and labels. `hashDataset()` computes a fast FNV-1a fingerprint over all texts and labels for data versioning. `compareExperiments()` diffs two runs showing accuracy delta, what changed (algorithm, data, features, hyperparams), and whether they used the same dataset. `bestExperiment()` returns the highest-accuracy run for a task. `experimentStats()` returns aggregate metrics across all runs.

Exports: `recordExperiment(entry)`, `listExperiments(taskName, opts)`, `getExperiment(id)`, `compareExperiments(idA, idB)`, `bestExperiment(taskName)`, `clearExperiments(taskName)`, `experimentStats(taskName)`, `hashDataset(rows)`, `openDb()`.

## lib/config.js

Loads project-level defaults from `distill.config.json` (optional). Supports: `model`, `batchSize`, `maxRetries`, `splitRatio`, `dataDir`, `modelsDir`. Values are merged with built-in defaults and cached for the session.

Exports: `loadConfig()`, `resetConfigCache()`, `DEFAULTS`.

## lib/log.js

Structured JSONL logging. Call `startLog(taskName)` at the start of a pipeline run, `logEntry(event, data)` throughout, and `flushLog()` at the end. Logs are written to `logs/<task-name>_<timestamp>.jsonl` with ISO timestamps on every entry.

Exports: `startLog(taskName)`, `logEntry(event, data)`, `flushLog()`.

## lib/templates.js

Loads pre-built task templates from the `templates/` directory. Ships with five templates: sentiment, intent, spam-detector, contact-extractor, and topic. Each template is a complete task definition that can be cloned and customized.

Exports: `listTemplates()`, `loadTemplate(name)`, `TEMPLATES_DIR`.

## lib/report.js

Generates standalone HTML evaluation reports with inline CSS. Includes a confusion matrix (color-coded by cell value), per-label precision/recall/F1/support table, label distribution bar chart, and grouped example errors. Reports are written to `reports/<task-name>_report.html`.

Exports: `generateReport(taskName, { valData, predictions, labels, meta })`, `confusionMatrix(actual, predicted, labels)`, `perLabelMetrics(actual, predicted, labels)`, `findErrors(data, predictions, { maxPerLabel })`.

## scripts/train.py

A standalone Python script that reads JSONL, trains a model, evaluates on validation data, and serializes the result. Uses `argparse` for CLI, `pickle` for model serialization, and scikit-learn for the ML pipeline. Supports `--onnx` for ONNX export and `--predict` for inference mode with confidence scores. Designed to be replaceable — any script that accepts the same flags will work.
