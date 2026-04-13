// Phase 10 — Evaluation and interpretability
// K-fold cross-validation, feature importance, error taxonomy,
// calibration analysis, data map coordinates.

import { join } from 'node:path'

const DATA_DIR = join(import.meta.dir, '..', 'data')
const MODELS_DIR = join(import.meta.dir, '..', 'models')

// ── K-Fold Cross-Validation ──────────────────────────────

// Split data into k folds, returns array of { train, val } index arrays
function kFoldSplit(n, k = 5) {
  const indices = Array.from({ length: n }, (_, i) => i)
  // Shuffle
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[indices[i], indices[j]] = [indices[j], indices[i]]
  }

  const foldSize = Math.floor(n / k)
  const folds = []
  for (let f = 0; f < k; f++) {
    const start = f * foldSize
    const end = f === k - 1 ? n : start + foldSize
    const valIndices = indices.slice(start, end)
    const trainIndices = [...indices.slice(0, start), ...indices.slice(end)]
    folds.push({ train: trainIndices, val: valIndices })
  }
  return folds
}

// Run k-fold cross-validation via Python subprocess
async function kFoldCV(task, data, { k = 5, algorithm = 'logistic_regression', onFold } = {}) {
  const { writeJsonl } = await import('./data.js')
  const { unlink } = await import('node:fs/promises')
  const folds = kFoldSplit(data.length, k)
  const scores = []

  for (let f = 0; f < folds.length; f++) {
    const fold = folds[f]
    const trainData = fold.train.map(i => data[i])
    const valData = fold.val.map(i => data[i])

    const trainPath = join(DATA_DIR, `${task.name}_cv_train_${f}.jsonl`)
    const valPath = join(DATA_DIR, `${task.name}_cv_val_${f}.jsonl`)
    await writeJsonl(trainPath, trainData)
    await writeJsonl(valPath, valData)

    const scriptPath = join(import.meta.dir, '..', 'scripts', 'train.py')
    const outputDir = join(MODELS_DIR, `${task.name}_cv_${f}`)

    const proc = Bun.spawn(['python3', scriptPath,
      '--train', trainPath, '--val', valPath, '--output', outputDir,
      '--task-type', task.type, '--labels', (task.labels || []).join(','),
      '--algorithm', algorithm
    ], { cwd: join(import.meta.dir, '..'), stdout: 'pipe', stderr: 'ignore' })

    const stdout = await new Response(proc.stdout).text()
    await proc.exited

    // Parse accuracy from output
    const accMatch = stdout.match(/Validation Accuracy:\s+([\d.]+)/)
    const acc = accMatch ? parseFloat(accMatch[1]) : 0
    scores.push(acc)

    onFold?.(f + 1, k, acc)

    // Clean up temp files
    try {
      await unlink(trainPath)
      await unlink(valPath)
    } catch {}
  }

  const mean = scores.reduce((a, b) => a + b, 0) / scores.length
  const variance = scores.reduce((a, b) => a + (b - mean) ** 2, 0) / scores.length
  const std = Math.sqrt(variance)

  return { scores, mean, std, k }
}

// ── Feature Importance ───────────────────────────────────

// Extract top TF-IDF features per label from a trained model
async function featureImportance(taskName, { topN = 10 } = {}) {
  // Use a small Python script to extract feature names and coefficients
  const script = `
import pickle, json, sys, os
model_path = sys.argv[1]
top_n = int(sys.argv[2])

with open(model_path, 'rb') as f:
    pipeline = pickle.load(f)

# Handle embedding models (dict or pipeline-only)
if isinstance(pipeline, dict):
    print(json.dumps({"error": "Feature importance not available for embedding models"}))
    sys.exit(0)

try:
    tfidf = pipeline.named_steps.get('tfidf')
    clf = pipeline.named_steps.get('clf')
except:
    print(json.dumps({"error": "Could not extract pipeline steps"}))
    sys.exit(0)

if tfidf is None or clf is None:
    print(json.dumps({"error": "Pipeline missing tfidf or clf step"}))
    sys.exit(0)

features = tfidf.get_feature_names_out()
result = {}

if hasattr(clf, 'coef_'):
    import numpy as np
    classes = list(clf.classes_) if hasattr(clf, 'classes_') else []
    for i, cls in enumerate(classes):
        coef = clf.coef_[i] if clf.coef_.ndim > 1 else clf.coef_[0]
        top_idx = np.argsort(coef)[-top_n:][::-1]
        result[str(cls)] = [{"feature": str(features[j]), "weight": float(coef[j])} for j in top_idx]
elif hasattr(clf, 'feature_importances_'):
    import numpy as np
    imp = clf.feature_importances_
    top_idx = np.argsort(imp)[-top_n:][::-1]
    result["global"] = [{"feature": str(features[j]), "weight": float(imp[j])} for j in top_idx]
else:
    result = {"error": "Classifier has no coef_ or feature_importances_"}

print(json.dumps(result))
`

  const modelPath = join(MODELS_DIR, taskName, 'model.pkl')
  const proc = Bun.spawn(['python3', '-c', script, modelPath, String(topN)], {
    cwd: join(import.meta.dir, '..'),
    stdout: 'pipe',
    stderr: 'ignore'
  })

  const stdout = await new Response(proc.stdout).text()
  await proc.exited

  return JSON.parse(stdout.trim())
}

// ── Error Taxonomy ───────────────────────────────────────

function errorTaxonomy(valData, predictions) {
  const taxonomy = {
    confusionPairs: {},    // Which label pairs are most confused
    byTextLength: { short: 0, medium: 0, long: 0 },  // Errors by text length
    byProvider: {},        // Errors by data source
    totalErrors: 0,
    totalCorrect: 0
  }

  for (let i = 0; i < valData.length; i++) {
    const actual = valData[i].label
    const predicted = predictions[i]?.label || predictions[i]
    const text = valData[i].text || ''

    if (actual === predicted) {
      taxonomy.totalCorrect++
      continue
    }

    taxonomy.totalErrors++

    // Confusion pairs
    const pair = `${actual} → ${predicted}`
    taxonomy.confusionPairs[pair] = (taxonomy.confusionPairs[pair] || 0) + 1

    // Text length
    if (text.length < 50) taxonomy.byTextLength.short++
    else if (text.length < 150) taxonomy.byTextLength.medium++
    else taxonomy.byTextLength.long++

    // Provider
    const provider = valData[i]._provider || valData[i]._source || 'unknown'
    taxonomy.byProvider[provider] = (taxonomy.byProvider[provider] || 0) + 1
  }

  // Sort confusion pairs by frequency
  taxonomy.topConfusions = Object.entries(taxonomy.confusionPairs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([pair, count]) => ({ pair, count }))

  return taxonomy
}

// ── Calibration Analysis ─────────────────────────────────

function calibrationBins(predictions, actual, { nBins = 10 } = {}) {
  // Only works with predictions that have confidence scores
  const withConf = predictions
    .map((p, i) => ({ confidence: p.confidence, correct: (p.label || p) === actual[i] }))
    .filter(p => p.confidence != null)

  if (!withConf.length) return { bins: [], ece: 0, hasConfidence: false }

  const bins = Array.from({ length: nBins }, () => ({ predictions: 0, correct: 0, avgConfidence: 0, avgAccuracy: 0 }))

  for (const p of withConf) {
    const binIdx = Math.min(Math.floor(p.confidence * nBins), nBins - 1)
    bins[binIdx].predictions++
    if (p.correct) bins[binIdx].correct++
  }

  // Compute averages and ECE
  let ece = 0
  for (let b = 0; b < bins.length; b++) {
    const bin = bins[b]
    if (bin.predictions > 0) {
      bin.avgAccuracy = bin.correct / bin.predictions
      bin.avgConfidence = (b + 0.5) / nBins
      ece += (bin.predictions / withConf.length) * Math.abs(bin.avgAccuracy - bin.avgConfidence)
    }
  }

  return { bins, ece, hasConfidence: true, total: withConf.length }
}

// ── Data Map Coordinates ─────────────────────────────────

// Generate 2D coordinates from embeddings using a simple PCA projection
// (Full UMAP/t-SNE would require Python — this gives a lightweight JS-only option)
function projectTo2D(embeddings) {
  if (!embeddings.length || !embeddings[0].length) return []

  const n = embeddings.length
  const d = embeddings[0].length

  // Compute mean
  const mean = new Array(d).fill(0)
  for (const emb of embeddings) {
    for (let j = 0; j < d; j++) mean[j] += emb[j]
  }
  for (let j = 0; j < d; j++) mean[j] /= n

  // Center data
  const centered = embeddings.map(emb => emb.map((v, j) => v - mean[j]))

  // Power iteration for top 2 principal components
  function powerIteration(matrix, nIter = 50) {
    let v = Array.from({ length: d }, () => Math.random() - 0.5)
    for (let iter = 0; iter < nIter; iter++) {
      const Av = new Array(d).fill(0)
      for (const row of matrix) {
        const dot = row.reduce((s, val, j) => s + val * v[j], 0)
        for (let j = 0; j < d; j++) Av[j] += dot * row[j]
      }
      const norm = Math.sqrt(Av.reduce((s, x) => s + x * x, 0)) || 1
      v = Av.map(x => x / norm)
    }
    return v
  }

  // Project onto first principal component
  const pc1 = powerIteration(centered)
  const proj1 = centered.map(row => row.reduce((s, v, j) => s + v * pc1[j], 0))

  // Deflate and get second component
  const deflated = centered.map(row => {
    const dot = row.reduce((s, v, j) => s + v * pc1[j], 0)
    return row.map((v, j) => v - dot * pc1[j])
  })
  const pc2 = powerIteration(deflated)
  const proj2 = deflated.map(row => row.reduce((s, v, j) => s + v * pc2[j], 0))

  return proj1.map((x, i) => ({ x, y: proj2[i] }))
}

export {
  kFoldSplit, kFoldCV,
  featureImportance,
  errorTaxonomy,
  calibrationBins,
  projectTo2D
}
