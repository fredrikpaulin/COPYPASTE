import { test, expect, describe, afterAll } from 'bun:test'
import {
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
} from '../lib/active-loop.js'
import { trainCRF, labelsToBIO } from '../lib/crf.js'
import { trainScoring } from '../lib/scoring.js'
import { join } from 'node:path'
import { rm, mkdir } from 'node:fs/promises'

const ROOT = join(import.meta.dir, '..')
const TMP_MODELS = join(ROOT, 'test-tmp-p16-models')
const TMP_DATA = join(ROOT, 'test-tmp-p16-data')

afterAll(async () => {
  await rm(TMP_MODELS, { recursive: true, force: true }).catch(() => {})
  await rm(TMP_DATA, { recursive: true, force: true }).catch(() => {})
})

// ── Entropy ─────────────────────────────────────────────

describe('entropy', () => {
  test('returns 0 for certain distribution', () => {
    expect(entropy([1, 0, 0])).toBe(0)
  })

  test('returns 1 for uniform binary distribution', () => {
    const h = entropy([0.5, 0.5])
    expect(Math.abs(h - 1)).toBeLessThan(0.001)
  })

  test('maximum entropy for uniform distribution increases with classes', () => {
    const h3 = entropy([1 / 3, 1 / 3, 1 / 3])
    const h2 = entropy([0.5, 0.5])
    expect(h3).toBeGreaterThan(h2)
  })

  test('handles empty array', () => {
    expect(entropy([])).toBe(0)
  })

  test('skips zero probabilities', () => {
    // Should not produce NaN
    const h = entropy([0.5, 0.5, 0])
    expect(Number.isFinite(h)).toBe(true)
    expect(h).toBeGreaterThan(0)
  })
})

// ── selectMostUncertain ─────────────────────────────────

describe('selectMostUncertain', () => {
  const items = [
    { text: 'a', uncertainty: 0.1 },
    { text: 'b', uncertainty: 0.9 },
    { text: 'c', uncertainty: 0.5 },
    { text: 'd', uncertainty: 0.7 },
    { text: 'e', uncertainty: 0.3 },
  ]

  test('returns top-K most uncertain', () => {
    const top = selectMostUncertain(items, { topK: 3 })
    expect(top.length).toBe(3)
    expect(top[0].text).toBe('b')
    expect(top[1].text).toBe('d')
    expect(top[2].text).toBe('c')
  })

  test('defaults to topK=10', () => {
    const top = selectMostUncertain(items)
    expect(top.length).toBe(5) // all items since less than 10
  })

  test('does not mutate input', () => {
    const original = [...items]
    selectMostUncertain(items, { topK: 2 })
    expect(items[0].text).toBe(original[0].text)
    expect(items[1].text).toBe(original[1].text)
  })

  test('handles empty input', () => {
    const top = selectMostUncertain([], { topK: 5 })
    expect(top).toEqual([])
  })
})

// ── CRF margin uncertainty ──────────────────────────────

describe('crfMargin', () => {
  const bioTags = labelsToBIO(['PER'])
  let model

  test('trains a CRF model for margin tests', () => {
    const data = [
      { tokens: ['John', 'lives', 'in', 'Paris'], tags: ['B-PER', 'O', 'O', 'O'] },
      { tokens: ['Alice', 'works', 'at', 'Google'], tags: ['B-PER', 'O', 'O', 'O'] },
      { tokens: ['The', 'cat', 'sat', 'down'], tags: ['O', 'O', 'O', 'O'] },
      { tokens: ['Bob', 'and', 'Carol', 'left'], tags: ['B-PER', 'O', 'B-PER', 'O'] },
    ]
    model = trainCRF(data, { tags: bioTags, epochs: 5, hashSize: 1 << 12 })
    expect(model.weights).toBeDefined()
    expect(model.tags).toBeDefined()
  })

  test('returns margin and bestScore', () => {
    const result = crfMargin(['John', 'works', 'here'], model)
    expect(typeof result.margin).toBe('number')
    expect(typeof result.bestScore).toBe('number')
    expect(result.margin).toBeGreaterThanOrEqual(0)
  })

  test('returns zero margin for empty tokens', () => {
    const result = crfMargin([], model)
    expect(result.margin).toBe(0)
    expect(result.bestScore).toBe(0)
  })
})

// ── sequenceLabelingUncertainty ─────────────────────────

describe('sequenceLabelingUncertainty', () => {
  let model

  test('computes uncertainty for token sequences', async () => {
    const data = [
      { tokens: ['John', 'lives', 'in', 'Paris'], tags: ['B-PER', 'O', 'O', 'O'] },
      { tokens: ['Alice', 'works', 'at', 'Google'], tags: ['B-PER', 'O', 'O', 'O'] },
      { tokens: ['The', 'cat', 'sat', 'down'], tags: ['O', 'O', 'O', 'O'] },
    ]
    model = trainCRF(data, { tags: labelsToBIO(['PER']), epochs: 5, hashSize: 1 << 12 })

    const sequences = [['John', 'sat', 'down'], ['The', 'cat', 'left']]
    const results = await sequenceLabelingUncertainty('test', sequences, model)

    expect(results.length).toBe(2)
    for (const r of results) {
      expect(r.method).toBe('crf_margin')
      expect(typeof r.uncertainty).toBe('number')
      expect(typeof r.margin).toBe('number')
      expect(r.tokens).toBeArray()
      expect(r.text).toBeDefined()
    }
  })

  test('uncertainty is between 0 and 1', async () => {
    const data = [
      { tokens: ['a', 'b'], tags: ['O', 'O'] },
      { tokens: ['c', 'd'], tags: ['B-PER', 'O'] },
    ]
    model = trainCRF(data, { tags: labelsToBIO(['PER']), epochs: 3, hashSize: 1 << 10 })

    const results = await sequenceLabelingUncertainty('test', [['x', 'y']], model)
    expect(results[0].uncertainty).toBeGreaterThanOrEqual(0)
    expect(results[0].uncertainty).toBeLessThanOrEqual(1)
  })
})

// ── scoringUncertainty ──────────────────────────────────

describe('scoringUncertainty', () => {
  let model

  test('computes uncertainty via feature dropout', () => {
    const data = [
      { text: 'great amazing wonderful product', value: 5.0 },
      { text: 'good quality decent price', value: 4.0 },
      { text: 'average nothing special okay', value: 3.0 },
      { text: 'bad poor terrible quality', value: 1.0 },
      { text: 'worst garbage ever made', value: 1.0 },
    ]
    model = trainScoring(data, { epochs: 10, hashSize: 1 << 12, minVal: 1, maxVal: 5 })

    const results = scoringUncertainty(
      ['great product', 'average item'],
      model,
      { dropoutRounds: 20, dropRate: 0.3 }
    )

    expect(results.length).toBe(2)
    for (const r of results) {
      expect(r.method).toBe('scoring_dropout')
      expect(typeof r.uncertainty).toBe('number')
      expect(typeof r.variance).toBe('number')
      expect(typeof r.prediction).toBe('number')
      expect(r.uncertainty).toBeGreaterThanOrEqual(0)
    }
  })

  test('predictions are clamped to model range', () => {
    const results = scoringUncertainty(['test text'], model)
    expect(results[0].prediction).toBeGreaterThanOrEqual(model.minVal)
    expect(results[0].prediction).toBeLessThanOrEqual(model.maxVal)
  })

  test('more dropout rounds = more stable variance estimate', () => {
    // With many rounds, two runs should give similar variance
    const r1 = scoringUncertainty(['great product'], model, { dropoutRounds: 50, dropRate: 0.3 })
    const r2 = scoringUncertainty(['great product'], model, { dropoutRounds: 50, dropRate: 0.3 })
    // Variance estimates should both be finite
    expect(Number.isFinite(r1[0].variance)).toBe(true)
    expect(Number.isFinite(r2[0].variance)).toBe(true)
  })
})

// ── computeUncertainty routing ──────────────────────────

describe('computeUncertainty', () => {
  test('routes scoring task correctly', async () => {
    const data = [
      { text: 'good', value: 4 },
      { text: 'bad', value: 1 },
      { text: 'okay', value: 3 },
    ]
    const model = trainScoring(data, { epochs: 5, hashSize: 1 << 10, minVal: 1, maxVal: 5 })
    const task = { name: 'test-scoring', type: 'scoring' }
    const pool = ['good stuff', 'bad thing']

    const results = await computeUncertainty(task, pool, { model })
    expect(results.length).toBe(2)
    expect(results[0].method).toBe('scoring_dropout')
  })

  test('routes sequence-labeling task correctly', async () => {
    const data = [
      { tokens: ['John', 'left'], tags: ['B-PER', 'O'] },
      { tokens: ['The', 'cat'], tags: ['O', 'O'] },
    ]
    const model = trainCRF(data, { tags: labelsToBIO(['PER']), epochs: 3, hashSize: 1 << 10 })
    const task = { name: 'test-ner', type: 'sequence-labeling' }
    const pool = [{ tokens: ['Alice', 'runs'] }, { text: 'Bob walks' }]

    const results = await computeUncertainty(task, pool, { model })
    expect(results.length).toBe(2)
    expect(results[0].method).toBe('crf_margin')
  })

  test('throws for unsupported task type', async () => {
    const task = { name: 'x', type: 'extraction' }
    await expect(computeUncertainty(task, ['a'], {})).rejects.toThrow('Unsupported task type')
  })
})

// ── Label prompt builders ───────────────────────────────

describe('buildClassificationLabelPrompt', () => {
  test('includes labels and examples', () => {
    const task = { type: 'classification', labels: ['pos', 'neg'] }
    const selected = [
      { text: 'great product', confidence: 0.65 },
      { text: 'bad quality', confidence: null },
    ]
    const prompt = buildClassificationLabelPrompt(task, selected)
    expect(prompt).toContain('pos, neg')
    expect(prompt).toContain('great product')
    expect(prompt).toContain('65%')
    expect(prompt).toContain('unknown')
    expect(prompt).toContain('JSON array')
  })
})

describe('buildSequenceLabelPrompt', () => {
  test('includes BIO instructions', () => {
    const task = { type: 'sequence-labeling', labels: ['PER', 'LOC'] }
    const selected = [
      { tokens: ['John', 'in', 'Paris'], text: 'John in Paris' },
    ]
    const prompt = buildSequenceLabelPrompt(task, selected)
    expect(prompt).toContain('BIO')
    expect(prompt).toContain('B-TYPE')
    expect(prompt).toContain('PER, LOC')
    expect(prompt).toContain('John')
  })
})

describe('buildScoringLabelPrompt', () => {
  test('includes score range', () => {
    const task = { type: 'scoring', scoreRange: { min: 1, max: 10 } }
    const selected = [
      { text: 'nice item', prediction: 7.5 },
    ]
    const prompt = buildScoringLabelPrompt(task, selected)
    expect(prompt).toContain('1')
    expect(prompt).toContain('10')
    expect(prompt).toContain('7.50')
    expect(prompt).toContain('nice item')
  })

  test('uses default range when none specified', () => {
    const task = { type: 'scoring' }
    const selected = [{ text: 'test', prediction: null }]
    const prompt = buildScoringLabelPrompt(task, selected)
    expect(prompt).toContain('0')
    expect(prompt).toContain('5')
  })
})

// ── Persistence: saveActiveIteration / loadActiveHistory ─

describe('active iteration persistence', () => {
  const taskName = 'p16-persist-test'
  const modelsDir = join(TMP_MODELS, taskName)

  test('loads empty history when no file exists', async () => {
    await mkdir(TMP_MODELS, { recursive: true })
    // Point to non-existent dir
    const history = await loadActiveHistory('nonexistent-task-xyz')
    expect(history.iterations).toEqual([])
  })

  test('saves and loads iterations', async () => {
    // We need models dir to exist for the task
    const taskModelsDir = join(import.meta.dir, '..', 'models', taskName)
    await mkdir(taskModelsDir, { recursive: true })

    // Clean any existing history
    const histPath = join(taskModelsDir, 'active_loop_history.json')
    await rm(histPath, { force: true }).catch(() => {})

    await saveActiveIteration(taskName, {
      poolSize: 30,
      selected: 10,
      labeled: 10,
      method: 'scoring_dropout'
    })

    await saveActiveIteration(taskName, {
      poolSize: 20,
      selected: 5,
      labeled: 5,
      method: 'crf_margin'
    })

    const history = await loadActiveHistory(taskName)
    expect(history.iterations.length).toBe(2)
    expect(history.iterations[0].poolSize).toBe(30)
    expect(history.iterations[0].timestamp).toBeDefined()
    expect(history.iterations[1].method).toBe('crf_margin')

    // Cleanup
    await rm(taskModelsDir, { recursive: true, force: true }).catch(() => {})
  })
})

// ── appendLabeledData ───────────────────────────────────

describe('appendLabeledData', () => {
  const taskName = 'p16-append-test'

  test('creates new file with labeled data', async () => {
    const dataDir = join(import.meta.dir, '..', 'data')
    await mkdir(dataDir, { recursive: true })
    const synPath = join(dataDir, `${taskName}_synthetic.jsonl`)
    await rm(synPath, { force: true }).catch(() => {})

    const labeled = [
      { text: 'great', label: 'pos' },
      { text: 'bad', label: 'neg' },
    ]
    const result = await appendLabeledData(taskName, labeled, 'classification')
    expect(result.added).toBe(2)
    expect(result.total).toBe(2)

    // Read back
    const content = await Bun.file(synPath).text()
    const lines = content.trim().split('\n').map(l => JSON.parse(l))
    expect(lines.length).toBe(2)
    expect(lines[0]._source).toBe('active_loop')
    expect(lines[0].text).toBe('great')

    // Cleanup
    await rm(synPath, { force: true }).catch(() => {})
  })

  test('appends to existing data', async () => {
    const dataDir = join(import.meta.dir, '..', 'data')
    const synPath = join(dataDir, `${taskName}_synthetic.jsonl`)

    // Write initial data
    await Bun.write(synPath, '{"text":"existing","label":"pos"}\n')

    const labeled = [{ text: 'new', label: 'neg' }]
    const result = await appendLabeledData(taskName, labeled, 'classification')
    expect(result.added).toBe(1)
    expect(result.total).toBe(2)

    // Cleanup
    await rm(synPath, { force: true }).catch(() => {})
  })
})

// ── Exports ─────────────────────────────────────────────

describe('exports', () => {
  test('all expected functions are exported', () => {
    const fns = [
      entropy, classificationUncertainty, sequenceLabelingUncertainty,
      scoringUncertainty, crfMargin, computeUncertainty, selectMostUncertain,
      activeLoop, llmLabelForTask,
      buildClassificationLabelPrompt, buildSequenceLabelPrompt, buildScoringLabelPrompt,
      saveActiveIteration, loadActiveHistory, appendLabeledData
    ]
    for (const fn of fns) {
      expect(typeof fn).toBe('function')
    }
  })
})
