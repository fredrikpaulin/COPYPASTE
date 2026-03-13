import { test, expect, describe } from 'bun:test'
import {
  extractTextFeatures, hashFeature, featureVector, fnv1a,
  scoreText,
  trainScoring, predictScore, predictScoreBatch,
  evaluateScoring,
  saveScoringModel, loadScoringModel, hasScoringModel
} from '../lib/scoring.js'
import {
  buildSystemPrompt, buildBatchPrompt, validateExample
} from '../lib/generate.js'
import { join } from 'node:path'
import { rmSync, mkdirSync } from 'node:fs'

// ── Feature extraction ───────────────────────────────────

describe('extractTextFeatures', () => {
  test('returns unigrams', () => {
    const feats = extractTextFeatures('hello world')
    expect(feats).toContain('w=hello')
    expect(feats).toContain('w=world')
  })

  test('returns bigrams', () => {
    const feats = extractTextFeatures('the quick brown')
    expect(feats).toContain('bi=the|quick')
    expect(feats).toContain('bi=quick|brown')
  })

  test('returns character trigrams', () => {
    const feats = extractTextFeatures('abc')
    expect(feats).toContain('c3=abc')
  })

  test('returns length features', () => {
    const feats = extractTextFeatures('one two three')
    expect(feats).toContain('wc=3')
    expect(feats.some(f => f.startsWith('cc='))).toBe(true)
  })

  test('detects punctuation', () => {
    const feats = extractTextFeatures('wow! great!')
    expect(feats).toContain('has_excl')
    expect(feats.some(f => f.startsWith('punct='))).toBe(true)
  })

  test('detects digits', () => {
    const feats = extractTextFeatures('got 5 stars')
    expect(feats).toContain('has_digit')
  })

  test('detects question marks', () => {
    const feats = extractTextFeatures('is it good?')
    expect(feats).toContain('has_q')
  })

  test('lowercases words', () => {
    const feats = extractTextFeatures('Hello WORLD')
    expect(feats).toContain('w=hello')
    expect(feats).toContain('w=world')
  })
})

// ── FNV-1a hashing ───────────────────────────────────────

describe('fnv1a', () => {
  test('returns consistent hash', () => {
    expect(fnv1a('test')).toBe(fnv1a('test'))
  })

  test('different strings produce different hashes', () => {
    expect(fnv1a('abc')).not.toBe(fnv1a('xyz'))
  })
})

// ── Hash feature ─────────────────────────────────────────

describe('hashFeature', () => {
  test('returns idx and sign', () => {
    const result = hashFeature('w=hello', 1024)
    expect(typeof result.idx).toBe('number')
    expect(result.idx).toBeGreaterThanOrEqual(0)
    expect(result.idx).toBeLessThan(1024)
    expect(Math.abs(result.sign)).toBe(1)
  })

  test('idx is within bounds', () => {
    for (const feat of ['a', 'b', 'c', 'long_feature_name_here']) {
      const { idx } = hashFeature(feat, 256)
      expect(idx).toBeGreaterThanOrEqual(0)
      expect(idx).toBeLessThan(256)
    }
  })
})

// ── Feature vector ───────────────────────────────────────

describe('featureVector', () => {
  test('creates Float64Array of correct size', () => {
    const vec = featureVector(['w=hello', 'w=world'], 512)
    expect(vec).toBeInstanceOf(Float64Array)
    expect(vec.length).toBe(512)
  })

  test('has non-zero entries', () => {
    const vec = featureVector(['w=test'], 512)
    const nonZero = Array.from(vec).filter(v => v !== 0)
    expect(nonZero.length).toBeGreaterThan(0)
  })
})

// ── Score text ───────────────────────────────────────────

describe('scoreText', () => {
  test('returns 0 for zero weights', () => {
    const weights = new Float64Array(256)
    expect(scoreText(['w=hello'], weights, 256)).toBe(0)
  })

  test('returns non-zero for trained weights', () => {
    const weights = new Float64Array(256)
    // Set some weights
    const { idx, sign } = hashFeature('w=good', 256)
    weights[idx] = 2.0
    const score = scoreText(['w=good'], weights, 256)
    expect(score).toBe(2.0 * sign)
  })
})

// ── Training ─────────────────────────────────────────────

describe('trainScoring', () => {
  const data = [
    { text: 'amazing wonderful excellent perfect', value: 5.0 },
    { text: 'great good nice solid quality', value: 4.0 },
    { text: 'okay average mediocre passable', value: 3.0 },
    { text: 'bad poor disappointing weak', value: 2.0 },
    { text: 'terrible awful horrible worst', value: 1.0 },
    { text: 'really excellent superb outstanding', value: 4.8 },
    { text: 'very bad extremely poor awful', value: 1.2 },
    { text: 'decent acceptable fair enough', value: 3.2 },
    { text: 'absolutely stunning incredible', value: 5.0 },
    { text: 'complete garbage waste of money', value: 1.0 }
  ]

  test('returns model with correct shape', () => {
    const model = trainScoring(data, { epochs: 5, hashSize: 1024 })
    expect(model.weights).toBeInstanceOf(Float64Array)
    expect(model.weights.length).toBe(1024)
    expect(model.hashSize).toBe(1024)
    expect(typeof model.minVal).toBe('number')
    expect(typeof model.maxVal).toBe('number')
  })

  test('reports onEpoch callback', () => {
    const epochs = []
    trainScoring(data, {
      epochs: 3,
      hashSize: 512,
      onEpoch: info => epochs.push(info)
    })
    expect(epochs.length).toBe(3)
    expect(epochs[0].epoch).toBe(1)
    expect(epochs[2].epoch).toBe(3)
    expect(typeof epochs[0].mse).toBe('number')
  })

  test('MSE decreases over training', () => {
    const mses = []
    trainScoring(data, {
      epochs: 20,
      hashSize: 1024,
      onEpoch: ({ mse }) => mses.push(mse)
    })
    // First epoch MSE should be higher than last
    expect(mses[0]).toBeGreaterThan(mses[mses.length - 1])
  })

  test('auto-detects min/max from data', () => {
    const model = trainScoring(data, { epochs: 1, hashSize: 256 })
    expect(model.minVal).toBe(1.0)
    expect(model.maxVal).toBe(5.0)
  })

  test('respects custom min/max', () => {
    const model = trainScoring(data, { epochs: 1, hashSize: 256, minVal: 0, maxVal: 10 })
    expect(model.minVal).toBe(0)
    expect(model.maxVal).toBe(10)
  })
})

// ── Prediction ───────────────────────────────────────────

describe('predictScore', () => {
  const data = [
    { text: 'amazing wonderful excellent', value: 5.0 },
    { text: 'great good nice', value: 4.0 },
    { text: 'okay average', value: 3.0 },
    { text: 'bad poor disappointing', value: 2.0 },
    { text: 'terrible awful horrible', value: 1.0 }
  ]

  test('returns a number', () => {
    const model = trainScoring(data, { epochs: 10, hashSize: 512 })
    const score = predictScore('pretty good product', model)
    expect(typeof score).toBe('number')
    expect(isNaN(score)).toBe(false)
  })

  test('clamps to range', () => {
    const model = trainScoring(data, { epochs: 10, hashSize: 512, minVal: 1, maxVal: 5 })
    const score = predictScore('some text', model)
    expect(score).toBeGreaterThanOrEqual(1)
    expect(score).toBeLessThanOrEqual(5)
  })
})

describe('predictScoreBatch', () => {
  test('returns array of predictions', () => {
    const data = [
      { text: 'good', value: 4.0 },
      { text: 'bad', value: 1.0 }
    ]
    const model = trainScoring(data, { epochs: 5, hashSize: 256 })
    const results = predictScoreBatch(['hello', 'world'], model)
    expect(results.length).toBe(2)
    expect(results[0]).toHaveProperty('text')
    expect(results[0]).toHaveProperty('score')
    expect(typeof results[0].score).toBe('number')
  })
})

// ── Evaluation ───────────────────────────────────────────

describe('evaluateScoring', () => {
  test('perfect predictions give zero error', () => {
    const data = [{ value: 3.0 }, { value: 4.0 }]
    const preds = [{ score: 3.0 }, { score: 4.0 }]
    const result = evaluateScoring(data, preds)
    expect(result.mse).toBe(0)
    expect(result.mae).toBe(0)
    expect(result.rmse).toBe(0)
    expect(result.correlation).toBeCloseTo(1.0, 5)
    expect(result.r2).toBeCloseTo(1.0, 5)
  })

  test('computes MSE correctly', () => {
    const data = [{ value: 1.0 }, { value: 2.0 }]
    const preds = [{ score: 2.0 }, { score: 4.0 }]
    const result = evaluateScoring(data, preds)
    // errors: 1, 2 => squared: 1, 4 => mse: 2.5
    expect(result.mse).toBe(2.5)
    expect(result.mae).toBe(1.5)
    expect(result.rmse).toBeCloseTo(Math.sqrt(2.5), 5)
  })

  test('computes MAE correctly', () => {
    const data = [{ value: 3.0 }, { value: 5.0 }]
    const preds = [{ score: 2.0 }, { score: 6.0 }]
    const result = evaluateScoring(data, preds)
    expect(result.mae).toBe(1.0)
  })

  test('correlation is 1 for perfectly correlated', () => {
    const data = [{ value: 1 }, { value: 2 }, { value: 3 }]
    const preds = [{ score: 10 }, { score: 20 }, { score: 30 }]
    const result = evaluateScoring(data, preds)
    expect(result.correlation).toBeCloseTo(1.0, 5)
  })

  test('correlation is -1 for inversely correlated', () => {
    const data = [{ value: 1 }, { value: 2 }, { value: 3 }]
    const preds = [{ score: 30 }, { score: 20 }, { score: 10 }]
    const result = evaluateScoring(data, preds)
    expect(result.correlation).toBeCloseTo(-1.0, 5)
  })

  test('handles empty data', () => {
    const result = evaluateScoring([], [])
    expect(result.n).toBe(0)
    expect(result.mse).toBe(0)
  })

  test('returns n', () => {
    const data = [{ value: 1 }, { value: 2 }, { value: 3 }]
    const preds = [{ score: 1 }, { score: 2 }, { score: 3 }]
    expect(evaluateScoring(data, preds).n).toBe(3)
  })
})

// ── Model persistence ────────────────────────────────────

describe('model persistence', () => {
  const tmpDir = join(import.meta.dir, '..', 'models', '__test_scoring__')

  test('save and load roundtrip', async () => {
    const data = [
      { text: 'good', value: 4.0 },
      { text: 'bad', value: 1.0 }
    ]
    const model = trainScoring(data, { epochs: 3, hashSize: 256, minVal: 0, maxVal: 5 })

    await saveScoringModel('__test_scoring__', model)

    const loaded = await loadScoringModel('__test_scoring__')
    expect(loaded).not.toBeNull()
    expect(loaded.hashSize).toBe(256)
    expect(loaded.minVal).toBe(0)
    expect(loaded.maxVal).toBe(5)
    expect(loaded.weights.length).toBe(256)

    // Predictions should match
    const origScore = predictScore('good product', model)
    const loadedScore = predictScore('good product', loaded)
    expect(origScore).toBeCloseTo(loadedScore, 10)

    // Cleanup
    try { rmSync(tmpDir, { recursive: true }) } catch {}
  })

  test('hasScoringModel returns true after save', async () => {
    const data = [{ text: 'a', value: 1.0 }]
    const model = trainScoring(data, { epochs: 1, hashSize: 64 })
    await saveScoringModel('__test_scoring__', model)

    const has = await hasScoringModel('__test_scoring__')
    expect(has).toBe(true)

    try { rmSync(tmpDir, { recursive: true }) } catch {}
  })

  test('loadScoringModel returns null for missing model', async () => {
    const loaded = await loadScoringModel('__nonexistent_scoring_model__')
    expect(loaded).toBeNull()
  })

  test('hasScoringModel returns false for missing model', async () => {
    const has = await hasScoringModel('__nonexistent_scoring_model__')
    expect(has).toBe(false)
  })
})

// ── generate.js integration ──────────────────────────────

describe('generate.js scoring support', () => {
  const scoringTask = {
    name: 'test-scoring',
    type: 'scoring',
    description: 'Score reviews 1-5',
    scoreRange: { min: 1, max: 5 },
    labels: undefined,
    fields: undefined,
    synthetic: { count: 10, prompt: 'Generate reviews', batchSize: 5 }
  }

  test('buildSystemPrompt includes score range', () => {
    const prompt = buildSystemPrompt(scoringTask)
    expect(prompt).toContain('Score range: 1 to 5')
    expect(prompt).toContain('scoring')
  })

  test('buildBatchPrompt includes scoring instructions', () => {
    const prompt = buildBatchPrompt(scoringTask, 5)
    expect(prompt).toContain('value')
    expect(prompt).toContain('number between 1 and 5')
  })

  test('buildSystemPrompt works for regression type too', () => {
    const regTask = { ...scoringTask, type: 'regression' }
    const prompt = buildSystemPrompt(regTask)
    expect(prompt).toContain('Score range')
  })

  test('validateExample accepts valid scoring example', () => {
    expect(validateExample({ text: 'good product', value: 4.5 }, scoringTask)).toBe(true)
  })

  test('validateExample rejects non-numeric value', () => {
    expect(validateExample({ text: 'test', value: 'high' }, scoringTask)).toBe(false)
  })

  test('validateExample rejects NaN value', () => {
    expect(validateExample({ text: 'test', value: NaN }, scoringTask)).toBe(false)
  })

  test('validateExample rejects out-of-range value', () => {
    expect(validateExample({ text: 'test', value: 6.0 }, scoringTask)).toBe(false)
    expect(validateExample({ text: 'test', value: 0.5 }, scoringTask)).toBe(false)
  })

  test('validateExample accepts edge values at boundaries', () => {
    expect(validateExample({ text: 'test', value: 1.0 }, scoringTask)).toBe(true)
    expect(validateExample({ text: 'test', value: 5.0 }, scoringTask)).toBe(true)
  })

  test('validateExample accepts regression type without range', () => {
    const noRange = { name: 'test', type: 'regression', description: 'test' }
    expect(validateExample({ text: 'test', value: 42 }, noRange)).toBe(true)
  })

  test('validateExample rejects missing text', () => {
    expect(validateExample({ value: 3.0 }, scoringTask)).toBe(false)
  })
})

// ── Schema & template ────────────────────────────────────

describe('schema and template', () => {
  test('schema includes scoring type', async () => {
    const schema = await Bun.file(join(import.meta.dir, '..', 'schemas', 'task.schema.json')).json()
    expect(schema.properties.type.enum).toContain('scoring')
  })

  test('schema has scoreRange property', async () => {
    const schema = await Bun.file(join(import.meta.dir, '..', 'schemas', 'task.schema.json')).json()
    expect(schema.properties.scoreRange).toBeDefined()
    expect(schema.properties.scoreRange.properties.min).toBeDefined()
    expect(schema.properties.scoreRange.properties.max).toBeDefined()
  })

  test('review-scorer template exists and is valid', async () => {
    const template = await Bun.file(join(import.meta.dir, '..', 'templates', 'review-scorer.json')).json()
    expect(template.name).toBe('review-scorer')
    expect(template.type).toBe('scoring')
    expect(template.scoreRange.min).toBe(1)
    expect(template.scoreRange.max).toBe(5)
    expect(template.synthetic.count).toBe(200)
  })
})

// ── Exports ──────────────────────────────────────────────

describe('exports', () => {
  test('lib/scoring.js exports all expected functions', () => {
    expect(typeof extractTextFeatures).toBe('function')
    expect(typeof hashFeature).toBe('function')
    expect(typeof featureVector).toBe('function')
    expect(typeof fnv1a).toBe('function')
    expect(typeof scoreText).toBe('function')
    expect(typeof trainScoring).toBe('function')
    expect(typeof predictScore).toBe('function')
    expect(typeof predictScoreBatch).toBe('function')
    expect(typeof evaluateScoring).toBe('function')
    expect(typeof saveScoringModel).toBe('function')
    expect(typeof loadScoringModel).toBe('function')
    expect(typeof hasScoringModel).toBe('function')
  })
})
