# Roadmap

Current status: **v0.1.8 — phases 1–12 complete.** All planned phases are implemented: foundation, production hardening, data quality, model capabilities, deployment, active learning, multi-provider extensibility, embedding-based models, curriculum learning, transfer learning, interpretability, ensemble inference, experiment tracking, and transformer distillation.

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

---

## Phase 7 — Embedding-based models ✅

Move beyond bag-of-words. TF-IDF works well when vocabulary carries the signal, but many tasks need semantic understanding.

### Sentence embeddings
Add a training path that uses sentence-transformer embeddings as features instead of TF-IDF. Support local embedding models via Ollama and API-based embeddings via OpenAI (`text-embedding-3-small`) and Anthropic. The embedding step slots into the existing sklearn pipeline — swap the vectorizer, keep the classifier.

### Embedding cache
Embedding the same text repeatedly is wasteful. Cache embeddings in a local SQLite database keyed by (text, model) so re-training and experimentation don't re-embed the entire dataset every run.

### Semantic deduplication
Replace trigram Jaccard similarity with cosine similarity on embeddings for fuzzy dedup. This catches paraphrases that share meaning but not vocabulary — "The food was great" and "Excellent meal" — which trigram matching misses entirely.

### Dimensionality reduction
Add optional PCA or UMAP reduction after embedding, configurable via task definition. Useful when embedding dimensions are high relative to dataset size, and for visualization in the evaluation report.

---

## Phase 8 — Curriculum and data strategy ✅

Control the order and composition of training data for better model quality.

### Curriculum learning
Train in stages — easy examples first, hard examples later. Score example difficulty using model confidence from a preliminary training pass, then sort the training set and present it in ascending difficulty order. This is especially effective for noisy synthetic data where some examples are borderline.

### LLM-as-judge quality scoring
Before training, score every generated example on a rubric (relevance, naturalness, label correctness) using the LLM. Store scores as metadata. Let the user set a quality threshold — only examples above the threshold enter the training set. More targeted than the existing confidence filter.

### Contrastive example generation
Generate hard negatives alongside positives. For each label, ask the LLM to produce examples that are easily confused with a different label — "almost positive but actually neutral." These adversarial examples improve decision boundaries more than random generation.

### Cross-provider ensembling
Generate data from multiple providers (e.g. Claude + GPT-4o + Llama3) and combine them. Different models have different biases in what they generate — mixing providers increases diversity. Track provenance so the user can see which provider contributed which examples.

---

## Phase 9 — Multi-task and transfer learning ✅

Train models that share knowledge across tasks.

### Shared feature extraction
When multiple tasks operate on similar text (e.g. sentiment and intent on customer messages), train a shared TF-IDF or embedding layer and task-specific classification heads. This improves sample efficiency — a sentiment model's text features help the intent model and vice versa.

### Fine-tuning pipeline
Add a training backend that fine-tunes a small transformer (e.g. DistilBERT, TinyBERT) instead of training a classical ML pipeline. Use HuggingFace Transformers as the training backend with the existing script-swap interface. Output a quantized ONNX model for fast inference.

### Zero-shot bootstrap
Before generating synthetic data, evaluate the task's zero-shot performance using the LLM directly on validation data. This establishes a baseline and lets the user see how much distillation actually helps. If zero-shot accuracy is already high enough, skip training entirely.

### Progressive distillation
Chain distillation steps: large model → medium model → small model. First distill Claude into a 7B Ollama model, then distill that into a classical ML model. Each step trades capability for speed, and the intermediate model can generate more data faster than the large one.

---

## Phase 10 — Evaluation and interpretability ✅

Understand why models make the decisions they do.

### K-fold cross-validation
Replace the single train/val split with k-fold cross-validation for more robust accuracy estimates. Report mean and standard deviation across folds. Especially important with smaller datasets where a single split can be misleading.

### Feature importance
For TF-IDF models, extract and display the top features (words/bigrams) per label. For embedding models, use SHAP or attention-based explanations. Surface these in the evaluation report so the user can verify the model is learning the right signals.

### Error taxonomy
Automatically categorize misclassifications — is the model confusing two specific labels? Are errors concentrated in short texts? Do examples from a specific provider fail more often? Group errors by pattern and present them in the report.

### Calibration analysis
Check whether the model's confidence scores are well-calibrated — when it says 90% confident, is it right 90% of the time? Add reliability diagrams to the evaluation report. Offer optional Platt scaling or isotonic regression to fix miscalibration.

### Data map visualization
Plot the training set on a 2D map (UMAP or t-SNE on embeddings) colored by label, with misclassified validation points highlighted. This gives an immediate visual sense of cluster quality, overlap between labels, and where the decision boundary is struggling.

---

## Phase 11 — Ensemble inference and experiment tracking ✅

Combine models and track what works.

### Ensemble inference
Train all three algorithms (logistic regression, SVM, random forest) side by side, then combine their predictions via weighted majority vote. Weight by validation accuracy so stronger models have more say. Report agreement ratio — how many models agree on the winning label.

### Confidence threshold
Add a rejection mechanism: predictions below a configurable confidence threshold return `rejected: true` instead of committing to a label. Useful in production where a "don't know" is better than a wrong answer.

### Experiment tracking
Log every training run in a SQLite database with: task, algorithm, accuracy, data fingerprint, feature mode, hyperparameters, duration. Compare experiments side by side — see what changed between two runs and whether accuracy improved. Query best-ever experiment per task.

---

## Phase 12 — Transformer distillation ✅

Fine-tune pretrained transformer models as an alternative to classical scikit-learn pipelines.

### Model presets
Five built-in presets — DistilBERT (66M), TinyBERT (14M), BERT-base (110M), RoBERTa (125M), MiniLM (33M) — plus support for any custom HuggingFace model name. Each preset has tuned defaults for learning rate, epochs, and sequence length.

### GPU support
Automatic device detection: CUDA (NVIDIA GPUs), MPS (Apple Silicon), or CPU fallback. FP16 mixed precision enabled on CUDA for faster training. Device info displayed in the TUI before training starts.

### Runtime dependencies
PyTorch and HuggingFace Transformers are optional — checked at runtime with clear install instructions if missing. scikit-learn remains the default backend; transformers are an upgrade path for when you need more capability.

### Training and evaluation
HuggingFace Trainer with per-epoch evaluation, structured JSON output for TUI progress tracking, and automatic experiment recording. ONNX export via `optimum` (preferred) or `torch.onnx` fallback.

### Prediction
Load fine-tuned transformer models for inference with softmax confidence scores. Same interface as classical prediction for consistency.
