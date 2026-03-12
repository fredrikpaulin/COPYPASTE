# Roadmap

Current status: **v0.1.2 — all phases complete.** Foundation, production hardening, data quality, model capabilities, deployment, active learning, and multi-provider extensibility are all implemented.

---

## Phase 1 — Production hardening ✅

Make the existing pipeline reliable enough for unattended use.

### Retries and rate limiting
The generator currently fails on the first API error. Add exponential backoff with jitter, configurable max retries, and respect for `retry-after` headers. Partial progress should be saved so a failed run can resume rather than restart.

### Input validation and error recovery
Validate generated examples against the expected schema before writing them. Drop malformed rows and log warnings rather than failing the entire batch. Surface a summary of dropped rows in the TUI.

### Logging
Write structured logs (JSONL) to `logs/<task-name>_<timestamp>.jsonl` for every pipeline run — API calls made, examples generated, errors encountered, training metrics. Useful for debugging and auditing.

### Configuration file
Support a project-level `distill.config.json` for defaults — API key source, default model, default batch size, data directory overrides. Reduce repetition across task definitions.

---

## Phase 2 — Data quality ✅

Improve the quality and diversity of generated training data.

### Prompt iteration tooling
Add a `preview` command that generates a small sample (5–10 examples) and displays them in the TUI for inspection before committing to a full generation run. Faster feedback loop for prompt tuning.

### Deduplication
Detect near-duplicate examples (exact match and fuzzy/embedding-based) and remove them before training. Synthetic data from LLMs tends toward repetition — this directly improves model quality.

### Data augmentation
Add simple text augmentations — synonym replacement, random insertion, back-translation stubs — to expand the effective training set from synthetic examples. Configurable per task.

### Label balance enforcement
Go beyond "roughly even" distribution. After generation, measure actual label counts and optionally run targeted follow-up batches for under-represented labels to hit a specified distribution.

### Confidence filtering
Score generated examples using Claude (or a second model) and filter out low-confidence or ambiguous examples before they enter the training set.

---

## Phase 3 — Model capabilities ✅

Expand what kinds of models the pipeline can train.

### Model comparison
Train multiple model types (logistic regression, SVM, random forest, small neural net) on the same data and compare validation metrics side-by-side in a TUI table. Pick the best one automatically or let the user choose.

### Extraction task training
Implement the extraction training path — currently a placeholder. Train per-field extractors or a simple span-based model for structured extraction tasks.

### Sequence labeling support
Add a proper sequence labeling training script (CRF or BiLSTM-CRF) for token-level tasks like NER.

### Hyperparameter search
Add a basic grid search or random search over model hyperparameters, reporting validation metrics for each configuration.

---

## Phase 4 — Deployment ✅

Close the loop from training to inference.

### ONNX export
Export trained models to ONNX format. This makes them usable from JavaScript via ONNX Runtime, closing the loop entirely — generate, train, and deploy without leaving the JS ecosystem.

### Inference module
Add a `lib/infer.js` module that loads a trained model (pickle via Python subprocess, or ONNX natively) and exposes a `predict(text)` function. Include a TUI screen for interactive testing — type text, see predictions.

### Model versioning
Track model versions in `models/<task-name>/` with timestamps and metadata. Support rollback to a previous version. Store the task definition snapshot that produced each model.

### Bundled deployment
Package a trained model + inference code as a standalone module that can be dropped into another project with zero configuration.

---

## Phase 5 — Active learning ✅

Use model uncertainty to drive smarter data generation.

### Uncertainty sampling
After training, run the model on a pool of unlabeled data (or generate candidate examples). Identify the examples where the model is least confident and surface them in the TUI.

### Human-in-the-loop labeling
Present uncertain examples in the TUI for manual labeling. Arrow keys to assign labels, batch save to the real data file. Mix these high-value human labels back into the training set.

### LLM-in-the-loop labeling
Send uncertain examples back to Claude for labeling instead of (or in addition to) human review. This creates a targeted generation loop that focuses API spend where it matters most.

### Iteration tracking
Track accuracy across training iterations. Show a history of model performance as the dataset grows through active learning cycles.

---

## Phase 6 — Multi-provider and extensibility ✅

Remove the hard dependency on a single LLM provider.

### Multi-provider generation
Support OpenAI, Mistral, and local models (Ollama) as generation backends alongside Claude. Configurable per task via a `provider` field. Abstract the API call behind a simple interface.

### Plugin training scripts
Formalize the training script contract as a documented interface. Publish example scripts for PyTorch, HuggingFace Transformers, and XGBoost so users can drop in alternatives.

### Task templates
Ship a library of pre-built task templates (sentiment analysis, intent classification, spam detection, NER, topic classification) that users can clone and customize rather than starting from scratch.

### Evaluation dashboard
Add a `report` command that generates an HTML evaluation report — confusion matrix, per-label metrics, example errors, dataset composition — viewable in a browser.
