# Changelog

## 0.1.13

### Few-shot prompt optimization (Phase 17)

- New `lib/few-shot.js` — select optimal few-shot examples for LLM prompts across all task types
- Four selection strategies: random (baseline), balanced (equal per class/bucket), diverse (greedy feature set-cover), similar (Jaccard n-gram similarity to query)
- `selectExamples()` unified interface routes to any strategy by name
- Text similarity via Jaccard over unigram + bigram sets (weighted 60/40)
- Feature extraction for diversity: word features, label/bucket, tag types, length bucket
- `formatExample()` and `buildFewShotPrompt()` — task-aware prompt formatting for classification, scoring, sequence-labeling, and extraction
- `evaluatePromptSet()` — score a few-shot set by running LLM inference on validation examples
- `optimizeFewShot()` — test multiple strategies, evaluate on held-out set, pick the best
- `saveFewShotConfig()` / `loadFewShotConfig()` — persist the winning strategy and examples
- Three new TUI menu items: "Few-shot prompt", "Few-shot optimize", "Few-shot config"
- Task menu expanded to 48 items (indices 0–47)

### Tests

- New `test/phase17.test.js` — tokenize (3), ngrams (3), jaccard (4), textSimilarity (4), exampleFeatures (5), selectRandom (3), selectBalanced (3), selectDiverse (2), selectSimilar (1), selectExamples (6), formatExample (4), taskHeader (3), buildFewShotPrompt (2), persistence (2), exports (1)
- 461 tests passing across 21 files

## 0.1.12

### Task-agnostic active learning loop (Phase 16)

- New `lib/active-loop.js` — unified active learning across all task types (classification, sequence-labeling, scoring)
- Three uncertainty measures: classification confidence (1 - softmax), Viterbi margin (CRF), feature dropout variance (scoring)
- `computeUncertainty()` routes to the correct measure based on `task.type`
- `selectMostUncertain()` ranks pool examples by uncertainty and returns top-K
- `activeLoop()` — full iteration: generate candidate pool → score uncertainty → select most uncertain → LLM labeling
- `llmLabelForTask()` builds task-type-specific prompts and parses JSON array responses
- Three prompt builders: `buildClassificationLabelPrompt`, `buildSequenceLabelPrompt`, `buildScoringLabelPrompt`
- `saveActiveIteration()` / `loadActiveHistory()` — persist iteration history as JSON in models directory
- `appendLabeledData()` — append active-loop-labeled examples to synthetic JSONL with `_source: 'active_loop'` tag
- Two new TUI menu items: "Active learning loop (any task type)", "Active loop history"
- Task menu expanded to 45 items (indices 0–44)

### Tests

- New `test/phase16.test.js` — entropy (5), selectMostUncertain (4), crfMargin (3), sequenceLabelingUncertainty (2), scoringUncertainty (3), computeUncertainty routing (3), label prompt builders (5), persistence (2), appendLabeledData (2), exports (1)
- 415 tests passing across 20 files

## 0.1.11

### Scoring tasks (Phase 15)

- New `scoring` task type for continuous-valued predictions (ratings, sentiment scores, toxicity, etc.)
- Pure JavaScript linear regressor with feature hashing — zero dependencies
- Feature extraction: unigrams, bigrams, character trigrams, length, punctuation density, capitalization ratio, digit/exclamation/question detection
- FNV-1a feature hashing with sign trick to fixed-size weight vector (default 2^16)
- SGD training with L2 regularization, learning rate decay, and configurable epochs
- Evaluation metrics: MSE, MAE, RMSE, Pearson correlation, R-squared
- Binary model persistence — Float64Array weights as raw buffer, metadata as JSON
- `generate.js` updated with scoring-specific system prompts (score range), batch prompts (value distribution), and validation (numeric, NaN, range checking)
- `scoreRange` property in task schema — configurable min/max for score clamping and validation
- `review-scorer` template: score product reviews 1.0–5.0
- Three new TUI screens: "Train scoring model", "Predict scoring (score text)", "Evaluate scoring (MSE/correlation)"
- Error distribution analysis in evaluation (< 0.5, 0.5–1.0, > 1.0 buckets)
- Worst predictions display for debugging
- Experiment recording with `algorithm: "scoring"` and `feature_mode: "scoring"`
- Task menu expanded to 43 items (indices 0–42)

### Tests

- New `test/phase15.test.js` — extractTextFeatures (8), fnv1a (2), hashFeature (2), featureVector (2), scoreText (2), trainScoring (5), predictScore (2), predictScoreBatch (1), evaluateScoring (7), model persistence (4), generate.js integration (10), schema/template (3), exports (1)
- 386 tests passing across 19 files

## 0.1.10

### CRF sequence labeling (Phase 14)

- Pure JavaScript CRF engine — no Python dependency, no external libraries
- Averaged structured perceptron training with configurable epochs and feature hashing
- Feature extraction: word identity, shape, prefix/suffix, capitalization, digit, hyphen, bigrams, prev/next word context, previous tag
- FNV-1a feature hashing to fixed-size weight vector — avoids unbounded feature dictionaries
- Viterbi decoding for optimal tag sequence prediction
- BIO tagging scheme: B-TYPE (begin), I-TYPE (inside), O (outside)
- `labelsToBIO()` converts entity labels to full BIO tag set, `validateBIO()` checks well-formedness
- Entity extraction from BIO tags with span tracking
- Entity-level evaluation: per-type precision/recall/F1, micro-averaged F1, token accuracy
- Binary model persistence — Float64Array weights saved as raw buffer, metadata as JSON
- `generate.js` updated with BIO-format prompt templates for LLM-based sequence data generation
- `validateExample()` extended to validate sequence-labeling examples (tokens/tags arrays, valid BIO tags)
- NER template added: PER, ORG, LOC entity types
- Three new TUI screens: "Train CRF", "Predict CRF (tag text)", "Evaluate CRF (entity F1)"
- Experiment recording with `algorithm: "crf"` and `feature_mode: "crf"`
- Task menu expanded to 40 items (indices 0-39)

### Tests

- New `test/phase14.test.js` — wordShape (2), extractFeatures (5), fnv1a (2), featureHash (1), labelsToBIO (1), validateBIO (3), extractEntities (4), viterbi (3), trainCRF (3), predictSequence (2), evaluateEntities (3), model persistence (4), generate.js integration (7), NER template (2), exports (1)
- 337 tests passing across 18 files

## 0.1.9

### Streaming generation (Phase 13)

- Stream tokens from all three LLM providers as they arrive — see generation output in real-time
- Anthropic SSE streaming — parse `content_block_delta` events for Claude responses
- OpenAI SSE streaming — parse `choices[0].delta.content` from streaming completions
- Ollama NDJSON streaming — parse newline-delimited JSON from local Ollama models
- `streamProvider()` — unified streaming interface with `onToken(token, fullText)` callback, retry logic, and error handling matching `callProvider()`
- SSE and NDJSON parsers handle chunked delivery, partial lines, and `[DONE]` sentinel
- `streamBox()` TUI component — renders streaming tokens with automatic line wrapping
- `generate()` and `preview()` accept `stream: true` + `onToken` callback
- Per-batch stream labels — shows "Batch 2/5" header when streaming multi-batch generation
- Two new menu items: "Preview (streaming)" and "Generate (streaming)"
- Task menu expanded to 37 items (indices 0-36)

### Tests

- New `test/phase13.test.js` — SSE parsing (5), NDJSON parsing (3), token extractors (8), streamProvider with mock servers (6), streamBox (3), generate streaming (2), export consistency (3)
- 294 tests passing across 17 files

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
