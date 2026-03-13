// Phase 11a — Ensemble inference and confidence thresholding
// Combine predictions from multiple algorithms, reject uncertain predictions.

import { join } from 'node:path'

const MODELS_DIR = join(import.meta.dir, '..', 'models')
const ALGORITHMS = ['logistic_regression', 'svm', 'random_forest']

// Train all algorithms and keep the model files side by side
// Returns { models: [{ algorithm, accuracy, modelDir }] }
async function trainEnsembleModels(task, trainPath, valPath, { onStdout, onStderr, onAlgorithm } = {}) {
  const { runTraining } = await import('./train.js')
  const { mkdir, cp, readdir } = await import('node:fs/promises')
  const models = []

  for (const algorithm of ALGORITHMS) {
    onAlgorithm?.(algorithm)

    const ensembleDir = join(MODELS_DIR, task.name, 'ensemble', algorithm)
    await mkdir(ensembleDir, { recursive: true })

    // Train to the main model dir, then copy into ensemble subdir
    const result = await runTraining(task, trainPath, valPath, {
      algorithm,
      onStdout,
      onStderr
    })

    // Copy artifacts into ensemble/<algorithm>/
    const modelDir = join(MODELS_DIR, task.name)
    const files = await readdir(modelDir)
    for (const file of files) {
      if (file === 'ensemble' || file === 'versions') continue
      await cp(join(modelDir, file), join(ensembleDir, file))
    }

    const meta = await Bun.file(join(ensembleDir, 'meta.json')).json()
    models.push({ algorithm, accuracy: meta.accuracy, modelDir: ensembleDir })
  }

  return { models }
}

// List trained ensemble models for a task
async function listEnsembleModels(taskName) {
  const { readdir } = await import('node:fs/promises')
  const ensembleDir = join(MODELS_DIR, taskName, 'ensemble')
  const models = []
  try {
    const dirs = await readdir(ensembleDir)
    for (const dir of dirs) {
      const metaFile = Bun.file(join(ensembleDir, dir, 'meta.json'))
      if (await metaFile.exists()) {
        const meta = await metaFile.json()
        models.push({ algorithm: dir, accuracy: meta.accuracy, ...meta })
      }
    }
  } catch {}
  return models
}

// Run prediction with a specific ensemble model
async function predictWithModel(taskName, algorithm, texts) {
  const { spawn } = await import('node:child_process')
  const modelPath = join(MODELS_DIR, taskName, 'ensemble', algorithm, 'model.pkl')
  const scriptPath = join(import.meta.dir, '..', 'scripts', 'train.py')

  return new Promise((resolve, reject) => {
    const proc = spawn('python3', [scriptPath, '--predict', modelPath, '--input', '-'], {
      cwd: join(import.meta.dir, '..'),
      env: { ...process.env }
    })

    for (const text of texts) proc.stdin.write(JSON.stringify({ text }) + '\n')
    proc.stdin.end()

    let stdout = ''
    proc.stdout.on('data', chunk => { stdout += chunk.toString() })
    proc.stderr.on('data', () => {})
    proc.on('close', code => {
      if (code === 0) {
        try { resolve(JSON.parse(stdout)) } catch { reject(new Error('Parse error')) }
      } else reject(new Error(`Predict failed (code ${code})`))
    })
    proc.on('error', reject)
  })
}

// Ensemble prediction — majority vote across all trained algorithms
// Returns array of { label, confidence, votes, rejected, perModel }
async function ensemblePredict(taskName, texts, { threshold = 0 } = {}) {
  const models = await listEnsembleModels(taskName)
  if (!models.length) throw new Error('No ensemble models found. Run "Train ensemble" first.')

  // Get predictions from each model
  const allPredictions = []
  for (const model of models) {
    const preds = await predictWithModel(taskName, model.algorithm, texts)
    allPredictions.push({ algorithm: model.algorithm, accuracy: model.accuracy, predictions: preds })
  }

  // Combine via weighted majority vote
  const results = texts.map((text, i) => {
    const votes = {}
    const perModel = []

    for (const { algorithm, accuracy, predictions } of allPredictions) {
      const pred = predictions[i]
      const label = pred.label || pred
      const conf = pred.confidence ?? null

      // Weight by validation accuracy
      votes[label] = (votes[label] || 0) + accuracy
      perModel.push({ algorithm, label, confidence: conf, weight: accuracy })
    }

    // Pick label with highest weighted vote
    const sorted = Object.entries(votes).sort((a, b) => b[1] - a[1])
    const topLabel = sorted[0][0]
    const totalWeight = Object.values(votes).reduce((a, b) => a + b, 0)
    const ensembleConfidence = sorted[0][1] / totalWeight

    // Agreement ratio — what fraction of models agree on the top label
    const agreeing = perModel.filter(m => m.label === topLabel).length
    const agreement = agreeing / perModel.length

    // Reject if below threshold
    const rejected = threshold > 0 && ensembleConfidence < threshold

    return {
      label: rejected ? null : topLabel,
      confidence: ensembleConfidence,
      agreement,
      rejected,
      votes: Object.fromEntries(sorted),
      perModel
    }
  })

  return results
}

// Predict with confidence threshold — returns { label, confidence, rejected }
// Uses the primary model (not ensemble), just adds rejection
async function predictWithThreshold(taskName, texts, { threshold = 0.5 } = {}) {
  const { predict } = await import('./infer.js')
  const predictions = await predict(taskName, texts)

  return predictions.map(p => ({
    ...p,
    rejected: p.confidence != null && p.confidence < threshold
  }))
}

export {
  trainEnsembleModels, listEnsembleModels,
  ensemblePredict, predictWithThreshold,
  ALGORITHMS
}
