// Pure JavaScript scoring/regression model.
// Trains a linear regressor with feature hashing for continuous-valued predictions.
// Zero dependencies — feature extraction, SGD training, evaluation all in JS.

import { join } from 'node:path'

const MODELS_DIR = join(import.meta.dir, '..', 'models')

// ── Feature extraction ───────────────────────────────────

function extractTextFeatures(text) {
  const lower = text.toLowerCase()
  const words = lower.split(/\s+/).filter(Boolean)
  const feats = []

  // Unigrams
  for (const w of words) feats.push(`w=${w}`)

  // Bigrams
  for (let i = 0; i < words.length - 1; i++) {
    feats.push(`bi=${words[i]}|${words[i + 1]}`)
  }

  // Character trigrams (first 200 chars)
  const trimmed = lower.slice(0, 200)
  for (let i = 0; i < trimmed.length - 2; i++) {
    feats.push(`c3=${trimmed.slice(i, i + 3)}`)
  }

  // Length features
  feats.push(`wc=${Math.min(words.length, 50)}`)
  feats.push(`cc=${Math.min(text.length, 500)}`)

  // Punctuation density
  const punctCount = (text.match(/[!?.,:;]/g) || []).length
  feats.push(`punct=${Math.min(punctCount, 20)}`)

  // Capitalization ratio
  const caps = (text.match(/[A-Z]/g) || []).length
  const capRatio = text.length > 0 ? Math.round(caps / text.length * 10) : 0
  feats.push(`capr=${capRatio}`)

  // Has numbers, has exclamation, has question
  if (/\d/.test(text)) feats.push('has_digit')
  if (text.includes('!')) feats.push('has_excl')
  if (text.includes('?')) feats.push('has_q')

  return feats
}

// ── Feature hashing ──────────────────────────────────────

function fnv1a(str) {
  let h = 0x811c9dc5
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h = (h * 0x01000193) >>> 0
  }
  return h
}

function hashFeature(feat, hashSize) {
  const h = fnv1a(feat)
  const idx = h % hashSize
  // Sign bit for hash trick (reduces collision impact)
  const sign = (h & 1) ? 1 : -1
  return { idx, sign }
}

function featureVector(features, hashSize) {
  const vec = new Float64Array(hashSize)
  for (const f of features) {
    const { idx, sign } = hashFeature(f, hashSize)
    vec[idx] += sign
  }
  return vec
}

// ── Scoring (dot product) ────────────────────────────────

function scoreText(features, weights, hashSize) {
  let s = 0
  for (const f of features) {
    const { idx, sign } = hashFeature(f, hashSize)
    s += weights[idx] * sign
  }
  return s
}

// ── Training (SGD with L2 regularization) ────────────────

function trainScoring(data, { epochs = 20, hashSize = (1 << 16), learningRate = 0.01, lambda = 0.001, minVal, maxVal, onEpoch } = {}) {
  const weights = new Float64Array(hashSize)

  // Auto-detect range if not provided
  if (minVal === undefined) minVal = Math.min(...data.map(d => d.value))
  if (maxVal === undefined) maxVal = Math.max(...data.map(d => d.value))

  for (let epoch = 0; epoch < epochs; epoch++) {
    // Shuffle
    const order = Array.from({ length: data.length }, (_, i) => i)
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]]
    }

    let totalLoss = 0

    for (const idx of order) {
      const { text, value } = data[idx]
      const feats = extractTextFeatures(text)
      const predicted = scoreText(feats, weights, hashSize)
      const error = predicted - value

      // SGD update: w -= lr * (error * feature + lambda * w)
      for (const f of feats) {
        const { idx: hi, sign } = hashFeature(f, hashSize)
        weights[hi] -= learningRate * (error * sign + lambda * weights[hi])
      }

      totalLoss += error * error
    }

    const mse = data.length > 0 ? totalLoss / data.length : 0
    onEpoch?.({ epoch: epoch + 1, epochs, mse })

    // Decay learning rate
    learningRate *= 0.95
  }

  return { weights, hashSize, minVal, maxVal }
}

// ── Prediction ───────────────────────────────────────────

function predictScore(text, model) {
  const { weights, hashSize, minVal, maxVal } = model
  const feats = extractTextFeatures(text)
  let raw = scoreText(feats, weights, hashSize)

  // Clamp to training range
  if (minVal !== undefined && maxVal !== undefined) {
    raw = Math.max(minVal, Math.min(maxVal, raw))
  }

  return raw
}

function predictScoreBatch(texts, model) {
  return texts.map(text => ({
    text,
    score: predictScore(text, model)
  }))
}

// ── Evaluation ───────────────────────────────────────────

function evaluateScoring(data, predictions) {
  const n = data.length
  if (n === 0) return { mse: 0, mae: 0, rmse: 0, correlation: 0, n: 0 }

  let sumSqErr = 0, sumAbsErr = 0
  let sumGold = 0, sumPred = 0
  const golds = [], preds = []

  for (let i = 0; i < n; i++) {
    const gold = data[i].value
    const pred = predictions[i].score
    const err = pred - gold

    sumSqErr += err * err
    sumAbsErr += Math.abs(err)
    sumGold += gold
    sumPred += pred
    golds.push(gold)
    preds.push(pred)
  }

  const mse = sumSqErr / n
  const mae = sumAbsErr / n
  const rmse = Math.sqrt(mse)

  // Pearson correlation
  const meanGold = sumGold / n
  const meanPred = sumPred / n
  let covGP = 0, varG = 0, varP = 0
  for (let i = 0; i < n; i++) {
    const dg = golds[i] - meanGold
    const dp = preds[i] - meanPred
    covGP += dg * dp
    varG += dg * dg
    varP += dp * dp
  }

  const correlation = (varG > 0 && varP > 0)
    ? covGP / (Math.sqrt(varG) * Math.sqrt(varP))
    : 0

  // R-squared
  const ssRes = sumSqErr
  const ssTot = varG
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0

  return { mse, mae, rmse, correlation, r2, n }
}

// ── Model persistence ────────────────────────────────────

async function saveScoringModel(taskName, model) {
  const dir = join(MODELS_DIR, taskName)
  await Bun.write(join(dir, 'scoring_weights.bin'), model.weights.buffer)
  await Bun.write(join(dir, 'scoring_meta.json'), JSON.stringify({
    hashSize: model.hashSize,
    minVal: model.minVal,
    maxVal: model.maxVal,
    algorithm: 'scoring',
    timestamp: new Date().toISOString()
  }, null, 2) + '\n')
  return dir
}

async function loadScoringModel(taskName) {
  const dir = join(MODELS_DIR, taskName)
  const metaFile = Bun.file(join(dir, 'scoring_meta.json'))
  if (!await metaFile.exists()) return null

  const meta = await metaFile.json()
  const buf = await Bun.file(join(dir, 'scoring_weights.bin')).arrayBuffer()
  const weights = new Float64Array(buf)

  return { weights, hashSize: meta.hashSize, minVal: meta.minVal, maxVal: meta.maxVal }
}

async function hasScoringModel(taskName) {
  const metaFile = Bun.file(join(MODELS_DIR, taskName, 'scoring_meta.json'))
  return metaFile.exists()
}

export {
  extractTextFeatures, hashFeature, featureVector, fnv1a,
  scoreText,
  trainScoring, predictScore, predictScoreBatch,
  evaluateScoring,
  saveScoringModel, loadScoringModel, hasScoringModel
}
