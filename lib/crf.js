// Pure JavaScript CRF (Conditional Random Field) for sequence labeling.
// Implements averaged structured perceptron with BIO tag features.
// No dependencies — feature hashing, Viterbi decoding, and training all in JS.

import { join } from 'node:path'

const MODELS_DIR = join(import.meta.dir, '..', 'models')

// ── Feature extraction ───────────────────────────────────

function wordShape(w) {
  return w.replace(/[A-Z]/g, 'X').replace(/[a-z]/g, 'x').replace(/\d/g, 'd')
    .replace(/(.)\1+/g, '$1') // collapse runs
}

function extractFeatures(tokens, i, prevTag) {
  const w = tokens[i]
  const lower = w.toLowerCase()
  const feats = [
    `w=${lower}`,
    `shape=${wordShape(w)}`,
    `pre2=${lower.slice(0, 2)}`,
    `pre3=${lower.slice(0, 3)}`,
    `suf2=${lower.slice(-2)}`,
    `suf3=${lower.slice(-3)}`,
    `len=${Math.min(w.length, 10)}`,
    `cap=${w[0] === w[0].toUpperCase() && w[0] !== w[0].toLowerCase() ? 'Y' : 'N'}`,
    `allcap=${w === w.toUpperCase() && w !== w.toLowerCase() ? 'Y' : 'N'}`,
    `digit=${/\d/.test(w) ? 'Y' : 'N'}`,
    `hyphen=${w.includes('-') ? 'Y' : 'N'}`,
    `prev_tag=${prevTag}`
  ]

  if (i > 0) {
    feats.push(`pw=${tokens[i - 1].toLowerCase()}`)
    feats.push(`bigram=${tokens[i - 1].toLowerCase()}|${lower}`)
  } else {
    feats.push('BOS')
  }

  if (i < tokens.length - 1) {
    feats.push(`nw=${tokens[i + 1].toLowerCase()}`)
  } else {
    feats.push('EOS')
  }

  if (i > 1) feats.push(`ppw=${tokens[i - 2].toLowerCase()}`)
  if (i < tokens.length - 2) feats.push(`nnw=${tokens[i + 2].toLowerCase()}`)

  return feats
}

// ── Feature hashing ──────────────────────────────────────

// FNV-1a hash to fixed-size index — avoids a growing feature dictionary
function fnv1a(str) {
  let h = 0x811c9dc5
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h = (h * 0x01000193) >>> 0
  }
  return h
}

function featureHash(feat, tag, hashSize) {
  return fnv1a(`${feat}:${tag}`) % hashSize
}

// ── Viterbi decoding ─────────────────────────────────────

function score(features, tag, weights, hashSize) {
  let s = 0
  for (const f of features) {
    s += weights[featureHash(f, tag, hashSize)] || 0
  }
  return s
}

function viterbi(tokens, tags, weights, hashSize) {
  const n = tokens.length
  if (n === 0) return { path: [], score: 0 }

  // dp[i][t] = best score ending at position i with tag t
  const dp = Array.from({ length: n }, () => new Float64Array(tags.length))
  const bp = Array.from({ length: n }, () => new Int32Array(tags.length))

  // Position 0 — features are the same for all tags at position 0 (prevTag='<S>')
  const feats0 = extractFeatures(tokens, 0, '<S>')
  for (let t = 0; t < tags.length; t++) {
    dp[0][t] = score(feats0, tags[t], weights, hashSize)
    bp[0][t] = -1
  }

  // Fill forward — cache features per (position, prevTag) since they don't depend on current tag
  for (let i = 1; i < n; i++) {
    // Pre-compute features for each possible previous tag at this position
    const featsByPrev = new Array(tags.length)
    for (let p = 0; p < tags.length; p++) {
      featsByPrev[p] = extractFeatures(tokens, i, tags[p])
    }

    for (let t = 0; t < tags.length; t++) {
      let best = -Infinity, bestPrev = 0
      for (let p = 0; p < tags.length; p++) {
        const s = dp[i - 1][p] + score(featsByPrev[p], tags[t], weights, hashSize)
        if (s > best) { best = s; bestPrev = p }
      }
      dp[i][t] = best
      bp[i][t] = bestPrev
    }
  }

  // Backtrack
  let bestEnd = 0
  for (let t = 1; t < tags.length; t++) {
    if (dp[n - 1][t] > dp[n - 1][bestEnd]) bestEnd = t
  }

  const path = new Array(n)
  path[n - 1] = tags[bestEnd]
  let cur = bestEnd
  for (let i = n - 2; i >= 0; i--) {
    cur = bp[i + 1][cur]
    path[i] = tags[cur]
  }

  return { path, score: dp[n - 1][bestEnd] }
}

// ── Training (averaged structured perceptron) ────────────

function trainCRF(data, { tags, epochs = 10, hashSize = (1 << 18), onEpoch } = {}) {
  const weights = new Float64Array(hashSize)
  const totals = new Float64Array(hashSize) // for averaging
  let updateCount = 0

  for (let epoch = 0; epoch < epochs; epoch++) {
    // Shuffle training data
    const order = Array.from({ length: data.length }, (_, i) => i)
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]]
    }

    let correct = 0, total = 0

    for (const idx of order) {
      const { tokens, tags: goldTags } = data[idx]
      const { path: predicted } = viterbi(tokens, tags, weights, hashSize)

      let allCorrect = true
      for (let i = 0; i < tokens.length; i++) {
        if (predicted[i] !== goldTags[i]) { allCorrect = false; break }
      }

      if (!allCorrect) {
        // Update: gold features += 1, predicted features -= 1
        for (let i = 0; i < tokens.length; i++) {
          const prevGold = i === 0 ? '<S>' : goldTags[i - 1]
          const prevPred = i === 0 ? '<S>' : predicted[i - 1]
          const goldFeats = extractFeatures(tokens, i, prevGold)
          const predFeats = extractFeatures(tokens, i, prevPred)

          for (const f of goldFeats) {
            const h = featureHash(f, goldTags[i], hashSize)
            weights[h] += 1
            totals[h] += updateCount
          }
          for (const f of predFeats) {
            const h = featureHash(f, predicted[i], hashSize)
            weights[h] -= 1
            totals[h] -= updateCount
          }
        }
      } else {
        correct++
      }
      total++
      updateCount++
    }

    const accuracy = total > 0 ? correct / total : 0
    onEpoch?.({ epoch: epoch + 1, epochs, accuracy, sequencesCorrect: correct, sequencesTotal: total })
  }

  // Average the weights
  const averaged = new Float64Array(hashSize)
  for (let i = 0; i < hashSize; i++) {
    averaged[i] = weights[i] - totals[i] / updateCount
  }

  return { weights: averaged, tags, hashSize }
}

// ── Prediction ───────────────────────────────────────────

function predictSequence(tokens, model) {
  const { weights, tags, hashSize } = model
  const { path } = viterbi(tokens, tags, weights, hashSize)
  return path
}

function predictBatch(sequences, model) {
  return sequences.map(tokens => ({
    tokens,
    tags: predictSequence(tokens, model)
  }))
}

// ── Evaluation ───────────────────────────────────────────

// Extract entities from BIO tags: [{type, start, end, text}]
function extractEntities(tokens, tags) {
  const entities = []
  let current = null

  for (let i = 0; i < tags.length; i++) {
    const tag = tags[i]
    if (tag.startsWith('B-')) {
      if (current) entities.push(current)
      current = { type: tag.slice(2), start: i, end: i + 1, tokens: [tokens[i]] }
    } else if (tag.startsWith('I-') && current && tag.slice(2) === current.type) {
      current.end = i + 1
      current.tokens.push(tokens[i])
    } else {
      if (current) { entities.push(current); current = null }
    }
  }
  if (current) entities.push(current)

  return entities.map(e => ({ ...e, text: e.tokens.join(' ') }))
}

// Entity-level precision/recall/F1
function evaluateEntities(goldData, predictions) {
  const byType = {}
  let totalTP = 0, totalFP = 0, totalFN = 0

  for (let i = 0; i < goldData.length; i++) {
    const gold = extractEntities(goldData[i].tokens, goldData[i].tags)
    const pred = extractEntities(predictions[i].tokens, predictions[i].tags)

    const goldSet = new Set(gold.map(e => `${e.type}:${e.start}:${e.end}`))
    const predSet = new Set(pred.map(e => `${e.type}:${e.start}:${e.end}`))

    // Collect all entity types
    for (const e of [...gold, ...pred]) {
      if (!byType[e.type]) byType[e.type] = { tp: 0, fp: 0, fn: 0 }
    }

    for (const key of predSet) {
      const type = key.split(':')[0]
      if (goldSet.has(key)) {
        byType[type].tp++
        totalTP++
      } else {
        byType[type].fp++
        totalFP++
      }
    }
    for (const key of goldSet) {
      const type = key.split(':')[0]
      if (!predSet.has(key)) {
        byType[type].fn++
        totalFN++
      }
    }
  }

  const results = {}
  for (const [type, { tp, fp, fn }] of Object.entries(byType)) {
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0
    const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0
    results[type] = { precision, recall, f1, support: tp + fn }
  }

  const microP = totalTP + totalFP > 0 ? totalTP / (totalTP + totalFP) : 0
  const microR = totalTP + totalFN > 0 ? totalTP / (totalTP + totalFN) : 0
  const microF1 = microP + microR > 0 ? 2 * microP * microR / (microP + microR) : 0

  // Token-level accuracy
  let tokenCorrect = 0, tokenTotal = 0
  for (let i = 0; i < goldData.length; i++) {
    for (let j = 0; j < goldData[i].tags.length; j++) {
      if (goldData[i].tags[j] === predictions[i].tags[j]) tokenCorrect++
      tokenTotal++
    }
  }

  return {
    byType: results,
    micro: { precision: microP, recall: microR, f1: microF1 },
    tokenAccuracy: tokenTotal > 0 ? tokenCorrect / tokenTotal : 0,
    totalEntities: { gold: totalTP + totalFN, predicted: totalTP + totalFP }
  }
}

// ── Model persistence ────────────────────────────────────

async function saveModel(taskName, model) {
  const dir = join(MODELS_DIR, taskName)
  await Bun.write(join(dir, 'crf_weights.bin'), model.weights.buffer)
  await Bun.write(join(dir, 'crf_meta.json'), JSON.stringify({
    tags: model.tags,
    hashSize: model.hashSize,
    algorithm: 'crf',
    timestamp: new Date().toISOString()
  }, null, 2) + '\n')
  return dir
}

async function loadModel(taskName) {
  const dir = join(MODELS_DIR, taskName)
  const metaFile = Bun.file(join(dir, 'crf_meta.json'))
  if (!await metaFile.exists()) return null

  const meta = await metaFile.json()
  const buf = await Bun.file(join(dir, 'crf_weights.bin')).arrayBuffer()
  const weights = new Float64Array(buf)

  return { weights, tags: meta.tags, hashSize: meta.hashSize }
}

async function hasCRFModel(taskName) {
  const metaFile = Bun.file(join(MODELS_DIR, taskName, 'crf_meta.json'))
  return metaFile.exists()
}

// ── BIO tag utilities ────────────────────────────────────

// Convert label set to BIO tag set: ["PER", "LOC"] → ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
function labelsToBIO(labels) {
  const tags = ['O']
  for (const label of labels) {
    tags.push(`B-${label}`, `I-${label}`)
  }
  return tags
}

// Validate that a tag sequence is well-formed BIO
function validateBIO(tags) {
  const errors = []
  for (let i = 0; i < tags.length; i++) {
    if (tags[i].startsWith('I-')) {
      const type = tags[i].slice(2)
      const prev = i > 0 ? tags[i - 1] : 'O'
      if (prev !== `B-${type}` && prev !== `I-${type}`) {
        errors.push({ position: i, tag: tags[i], message: `I-${type} without preceding B-${type}` })
      }
    }
  }
  return errors
}

export {
  extractFeatures, featureHash, fnv1a, wordShape,
  viterbi, score,
  trainCRF, predictSequence, predictBatch,
  extractEntities, evaluateEntities,
  saveModel, loadModel, hasCRFModel,
  labelsToBIO, validateBIO
}
