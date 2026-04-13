// Task-agnostic active learning loop.
// Measures model uncertainty across all task types and selects the most
// informative examples for LLM labeling, reducing API cost by only generating
// labels where the small model is least confident.
//
// Uncertainty measures:
//   classification — entropy of softmax confidence scores
//   sequence-labeling — Viterbi margin (best score minus second-best)
//   scoring — prediction instability via feature dropout

import { join } from 'node:path'
import { extractFeatures as crfExtractFeatures, viterbi, score as crfScore, loadModel as loadCRFModel } from './crf.js'
import { extractTextFeatures, scoreText, loadScoringModel } from './scoring.js'

const MODELS_DIR = join(import.meta.dir, '..', 'models')
const DATA_DIR = join(import.meta.dir, '..', 'data')

// ── Uncertainty: Classification ──────────────────────────

// Entropy of a probability distribution (higher = more uncertain)
function entropy(probs) {
  let h = 0
  for (const p of probs) {
    if (p > 0) h -= p * Math.log2(p)
  }
  return h
}

// Get classification predictions with confidence via Python subprocess
async function classificationUncertainty(taskName, texts) {
  const { predict } = await import('./infer.js')
  const predictions = await predict(taskName, texts)
  return predictions.map((p, i) => ({
    text: texts[i],
    prediction: p.label,
    confidence: p.confidence,
    uncertainty: p.confidence != null ? 1 - p.confidence : 1,
    method: 'classification_confidence'
  }))
}

// ── Uncertainty: Sequence labeling (CRF) ─────────────────

// Viterbi margin: difference between best and second-best path scores
// Low margin = model is uncertain between two tag sequences
function crfMargin(tokens, model) {
  const { weights, tags, hashSize } = model
  const n = tokens.length
  if (n === 0) return { margin: 0, bestScore: 0 }

  // Single DP pass to get all final tag scores (avoids running viterbi twice)
  const dp = Array.from({ length: n }, () => new Float64Array(tags.length))

  // Position 0
  const feats0 = crfExtractFeatures(tokens, 0, '<S>')
  for (let t = 0; t < tags.length; t++) {
    dp[0][t] = crfScore(feats0, tags[t], weights, hashSize)
  }
  // Fill forward with feature caching per prevTag
  for (let i = 1; i < n; i++) {
    const featsByPrev = new Array(tags.length)
    for (let p = 0; p < tags.length; p++) {
      featsByPrev[p] = crfExtractFeatures(tokens, i, tags[p])
    }
    for (let t = 0; t < tags.length; t++) {
      let bestPrev = -Infinity
      for (let p = 0; p < tags.length; p++) {
        const s = dp[i - 1][p] + crfScore(featsByPrev[p], tags[t], weights, hashSize)
        if (s > bestPrev) bestPrev = s
      }
      dp[i][t] = bestPrev
    }
  }

  // Get top two final scores
  const finalScores = Array.from(dp[n - 1]).sort((a, b) => b - a)
  const bestScore = finalScores[0]
  const margin = finalScores.length >= 2 ? finalScores[0] - finalScores[1] : Infinity

  return { margin, bestScore }
}

async function sequenceLabelingUncertainty(taskName, sequences, model) {
  return sequences.map((tokens, i) => {
    const { margin, bestScore } = crfMargin(tokens, model)
    return {
      text: tokens.join(' '),
      tokens,
      prediction: null, // Could add tag prediction here
      confidence: null,
      uncertainty: margin === Infinity ? 0 : 1 / (1 + margin), // Sigmoid-like: low margin → high uncertainty
      margin,
      method: 'crf_margin'
    }
  })
}

// ── Uncertainty: Scoring ─────────────────────────────────

// Feature dropout: predict multiple times with random feature subsets,
// measure variance. High variance = model depends on specific features = uncertain.
function scoringUncertainty(texts, model, { dropoutRounds = 10, dropRate = 0.3 } = {}) {
  const { weights, hashSize, minVal, maxVal } = model

  return texts.map(text => {
    const feats = extractTextFeatures(text)
    const fullScore = clamp(scoreText(feats, weights, hashSize), minVal, maxVal)

    // Run dropout rounds
    const scores = []
    for (let r = 0; r < dropoutRounds; r++) {
      const subset = feats.filter(() => Math.random() > dropRate)
      if (subset.length === 0) { scores.push(fullScore); continue }
      const s = clamp(scoreText(subset, weights, hashSize), minVal, maxVal)
      scores.push(s)
    }

    // Compute variance
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length
    const variance = scores.reduce((a, s) => a + (s - mean) ** 2, 0) / scores.length
    const std = Math.sqrt(variance)

    return {
      text,
      prediction: fullScore,
      confidence: null,
      uncertainty: std, // Higher std = more uncertain
      variance,
      method: 'scoring_dropout'
    }
  })
}

function clamp(v, min, max) {
  if (min !== undefined && max !== undefined) return Math.max(min, Math.min(max, v))
  return v
}

// ── Unified interface ────────────────────────────────────

// Compute uncertainty for a pool of unlabeled texts.
// Returns sorted by uncertainty (most uncertain first).
async function computeUncertainty(task, pool, { model } = {}) {
  const type = task.type

  if (type === 'classification') {
    return classificationUncertainty(task.name, pool.map(p => p.text || p))
  }

  if (type === 'sequence-labeling') {
    if (!model) {
      model = await loadCRFModel(task.name)
    }
    if (!model) throw new Error('No CRF model found')
    const sequences = pool.map(p => {
      if (Array.isArray(p.tokens)) return p.tokens
      if (typeof p === 'string') return p.split(/\s+/)
      return (p.text || '').split(/\s+/)
    })
    return sequenceLabelingUncertainty(task.name, sequences, model)
  }

  if (type === 'scoring' || type === 'regression') {
    if (!model) {
      model = await loadScoringModel(task.name)
    }
    if (!model) throw new Error('No scoring model found')
    const texts = pool.map(p => p.text || p)
    return scoringUncertainty(texts, model)
  }

  throw new Error(`Unsupported task type for active learning: ${type}`)
}

// Select the top-K most uncertain examples from a pool.
function selectMostUncertain(uncertainties, { topK = 10 } = {}) {
  const sorted = [...uncertainties].sort((a, b) => b.uncertainty - a.uncertainty)
  return sorted.slice(0, topK)
}

// ── Active learning iteration ────────────────────────────

// Full active learning loop iteration:
// 1. Generate candidate pool via LLM
// 2. Score uncertainty with current model
// 3. Select most uncertain
// 4. Send to LLM for labeling
// 5. Add labeled examples to training data
async function activeLoop(task, {
  apiKey,
  poolSize = 30,
  selectK = 10,
  model,
  onPool,
  onUncertainty,
  onLabeled,
  provider
} = {}) {
  // Step 1: Generate candidate pool
  const { preview } = await import('./generate.js')
  const candidates = await preview(task, { apiKey, count: poolSize, provider })
  if (!candidates.examples.length) {
    return { pool: [], selected: [], labeled: [], added: 0 }
  }
  onPool?.(candidates.examples.length)

  // Step 2: Compute uncertainty
  const uncertainties = await computeUncertainty(task, candidates.examples, { model })
  onUncertainty?.(uncertainties.length)

  // Step 3: Select most uncertain
  const selected = selectMostUncertain(uncertainties, { topK: selectK })

  // Step 4: LLM labeling
  const labeled = await llmLabelForTask(task, selected, { apiKey, provider })
  onLabeled?.(labeled.length)

  return {
    pool: candidates.examples,
    selected,
    labeled,
    poolSize: candidates.examples.length,
    dropped: candidates.dropped || 0
  }
}

// Task-type-aware LLM labeling
async function llmLabelForTask(task, selected, { apiKey, provider }) {
  const { callProvider } = await import('./provider.js')
  const providerName = provider || task.synthetic?.provider || 'anthropic'

  let prompt
  if (task.type === 'classification') {
    prompt = buildClassificationLabelPrompt(task, selected)
  } else if (task.type === 'sequence-labeling') {
    prompt = buildSequenceLabelPrompt(task, selected)
  } else if (task.type === 'scoring' || task.type === 'regression') {
    prompt = buildScoringLabelPrompt(task, selected)
  } else {
    throw new Error(`Unsupported task type for labeling: ${task.type}`)
  }

  const text = await callProvider(providerName, {
    apiKey,
    model: task.synthetic?.model,
    systemPrompt: `You are labeling training data for: ${task.description}`,
    userPrompt: prompt,
    maxRetries: 2
  })

  const match = text.match(/\[[\s\S]*\]/)
  if (!match) throw new Error('No JSON array in labeling response')
  return JSON.parse(match[0])
}

function buildClassificationLabelPrompt(task, selected) {
  const examples = selected.map((s, i) =>
    `${i + 1}. "${s.text}" (model confidence: ${s.confidence != null ? (s.confidence * 100).toFixed(0) + '%' : 'unknown'})`
  ).join('\n')

  return `Label these examples with one of: ${(task.labels || []).join(', ')}

${examples}

Respond with ONLY a JSON array: [{"text": "...", "label": "..."}]`
}

function buildSequenceLabelPrompt(task, selected) {
  const examples = selected.map((s, i) =>
    `${i + 1}. tokens: ${JSON.stringify(s.tokens || s.text.split(/\s+/))}`
  ).join('\n')

  return `Tag these token sequences using BIO scheme with entity types: ${(task.labels || []).join(', ')}
B-TYPE = first token of entity, I-TYPE = continuation, O = outside.

${examples}

Respond with ONLY a JSON array: [{"tokens": [...], "tags": [...]}]`
}

function buildScoringLabelPrompt(task, selected) {
  const range = task.scoreRange || { min: 0, max: 5 }
  const examples = selected.map((s, i) =>
    `${i + 1}. "${s.text}" (model predicted: ${s.prediction != null ? s.prediction.toFixed(2) : '?'})`
  ).join('\n')

  return `Score these texts from ${range.min} to ${range.max}:

${examples}

Respond with ONLY a JSON array: [{"text": "...", "value": <number>}]`
}

// ── Persist iteration ────────────────────────────────────

async function saveActiveIteration(taskName, iteration) {
  const historyPath = join(MODELS_DIR, taskName, 'active_loop_history.json')
  let history = { iterations: [] }
  const file = Bun.file(historyPath)
  if (await file.exists()) history = await file.json()

  history.iterations.push({
    ...iteration,
    timestamp: new Date().toISOString()
  })
  await Bun.write(historyPath, JSON.stringify(history, null, 2) + '\n')
  return history
}

async function loadActiveHistory(taskName) {
  const historyPath = join(MODELS_DIR, taskName, 'active_loop_history.json')
  const file = Bun.file(historyPath)
  if (await file.exists()) return file.json()
  return { iterations: [] }
}

// ── Append labeled data to training file ─────────────────

async function appendLabeledData(taskName, labeled, taskType) {
  const synPath = join(DATA_DIR, `${taskName}_synthetic.jsonl`)
  let existing = []
  try {
    const { readJsonl } = await import('./data.js')
    existing = await readJsonl(synPath)
  } catch {}

  const newExamples = labeled.map(l => ({
    ...l,
    _source: 'active_loop'
  }))

  const all = [...existing, ...newExamples]
  const lines = all.map(e => JSON.stringify(e)).join('\n') + '\n'
  await Bun.write(synPath, lines)

  return { path: synPath, added: newExamples.length, total: all.length }
}

export {
  entropy,
  classificationUncertainty,
  sequenceLabelingUncertainty,
  scoringUncertainty,
  crfMargin,
  computeUncertainty,
  selectMostUncertain,
  activeLoop,
  llmLabelForTask,
  buildClassificationLabelPrompt,
  buildSequenceLabelPrompt,
  buildScoringLabelPrompt,
  saveActiveIteration,
  loadActiveHistory,
  appendLabeledData
}
