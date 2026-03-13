import { test, expect, describe, afterAll } from 'bun:test'
import {
  tokenize, ngrams, jaccard, textSimilarity,
  exampleFeatures,
  selectRandom, selectBalanced, selectDiverse, selectSimilar,
  selectExamples,
  formatExample, buildFewShotPrompt, taskHeader,
  evaluatePromptSet, optimizeFewShot,
  saveFewShotConfig, loadFewShotConfig
} from '../lib/few-shot.js'
import { join } from 'node:path'
import { rm, mkdir } from 'node:fs/promises'

const ROOT = join(import.meta.dir, '..')

// ── Text similarity helpers ─────────────────────────────

describe('tokenize', () => {
  test('splits on whitespace and lowercases', () => {
    expect(tokenize('Hello World')).toEqual(['hello', 'world'])
  })

  test('filters empty tokens', () => {
    expect(tokenize('  a  b  ')).toEqual(['a', 'b'])
  })

  test('handles empty string', () => {
    expect(tokenize('')).toEqual([])
  })
})

describe('ngrams', () => {
  test('produces unigrams', () => {
    const g = ngrams(['a', 'b', 'c'], 1)
    expect(g.has('a')).toBe(true)
    expect(g.has('b')).toBe(true)
    expect(g.has('c')).toBe(true)
    expect(g.size).toBe(3)
  })

  test('produces bigrams', () => {
    const g = ngrams(['a', 'b', 'c'], 2)
    expect(g.has('a b')).toBe(true)
    expect(g.has('b c')).toBe(true)
    expect(g.size).toBe(2)
  })

  test('returns empty for n > length', () => {
    const g = ngrams(['a'], 2)
    expect(g.size).toBe(0)
  })
})

describe('jaccard', () => {
  test('identical sets = 1', () => {
    expect(jaccard(new Set([1, 2, 3]), new Set([1, 2, 3]))).toBe(1)
  })

  test('disjoint sets = 0', () => {
    expect(jaccard(new Set([1, 2]), new Set([3, 4]))).toBe(0)
  })

  test('partial overlap', () => {
    // {1,2,3} ∩ {2,3,4} = {2,3}, union = {1,2,3,4}
    expect(jaccard(new Set([1, 2, 3]), new Set([2, 3, 4]))).toBe(2 / 4)
  })

  test('both empty = 1', () => {
    expect(jaccard(new Set(), new Set())).toBe(1)
  })
})

describe('textSimilarity', () => {
  test('identical text = 1', () => {
    expect(textSimilarity('hello world', 'hello world')).toBe(1)
  })

  test('completely different text = 0', () => {
    expect(textSimilarity('hello world', 'foo bar baz')).toBe(0)
  })

  test('partial overlap gives intermediate value', () => {
    const sim = textSimilarity('the cat sat', 'the cat ran')
    expect(sim).toBeGreaterThan(0)
    expect(sim).toBeLessThan(1)
  })

  test('is case insensitive', () => {
    expect(textSimilarity('Hello World', 'hello world')).toBe(1)
  })
})

// ── Feature extraction ──────────────────────────────────

describe('exampleFeatures', () => {
  test('includes word features', () => {
    const feats = exampleFeatures({ text: 'good product' }, { type: 'classification' })
    expect(feats.has('w=good')).toBe(true)
    expect(feats.has('w=product')).toBe(true)
  })

  test('includes label for classification', () => {
    const feats = exampleFeatures({ text: 'test', label: 'pos' }, { type: 'classification' })
    expect(feats.has('label=pos')).toBe(true)
  })

  test('includes bucket for scoring', () => {
    const feats = exampleFeatures({ text: 'test', value: 4.3 }, { type: 'scoring' })
    expect(feats.has('bucket=4')).toBe(true)
  })

  test('includes tag features for sequence labeling', () => {
    const feats = exampleFeatures(
      { text: 'John left', tags: ['B-PER', 'O'] },
      { type: 'sequence-labeling' }
    )
    expect(feats.has('tag=B-PER')).toBe(true)
    expect(feats.has('tag=O')).toBe(true)
  })

  test('includes length bucket', () => {
    const feats = exampleFeatures({ text: 'a b c d e f g h i j k' }, { type: 'classification' })
    expect(feats.has('len=10')).toBe(true)
  })
})

// ── Selection: random ───────────────────────────────────

describe('selectRandom', () => {
  const items = Array.from({ length: 20 }, (_, i) => ({ text: `item ${i}`, label: `l${i}` }))

  test('returns k items', () => {
    const selected = selectRandom(items, { k: 5 })
    expect(selected.length).toBe(5)
  })

  test('returns all if k > length', () => {
    const selected = selectRandom(items, { k: 100 })
    expect(selected.length).toBe(20)
  })

  test('does not mutate input', () => {
    const copy = [...items]
    selectRandom(items, { k: 5 })
    expect(items[0]).toEqual(copy[0])
  })
})

// ── Selection: balanced ─────────────────────────────────

describe('selectBalanced', () => {
  const task = { type: 'classification', labels: ['pos', 'neg'] }
  const items = [
    ...Array.from({ length: 10 }, (_, i) => ({ text: `pos ${i}`, label: 'pos' })),
    ...Array.from({ length: 10 }, (_, i) => ({ text: `neg ${i}`, label: 'neg' })),
  ]

  test('selects equal per class', () => {
    const selected = selectBalanced(items, task, { k: 6 })
    expect(selected.length).toBe(6)
    const posCount = selected.filter(e => e.label === 'pos').length
    const negCount = selected.filter(e => e.label === 'neg').length
    expect(posCount).toBe(3)
    expect(negCount).toBe(3)
  })

  test('handles imbalanced data', () => {
    const imbalanced = [
      { text: 'a', label: 'pos' },
      { text: 'b', label: 'neg' },
      { text: 'c', label: 'neg' },
      { text: 'd', label: 'neg' },
    ]
    const selected = selectBalanced(imbalanced, task, { k: 3 })
    expect(selected.length).toBe(3)
    const posCount = selected.filter(e => e.label === 'pos').length
    expect(posCount).toBeGreaterThanOrEqual(1)
  })

  test('works for scoring tasks', () => {
    const scoringTask = { type: 'scoring' }
    const scoringItems = [
      { text: 'a', value: 1.2 },
      { text: 'b', value: 2.8 },
      { text: 'c', value: 3.1 },
      { text: 'd', value: 1.4 },
    ]
    const selected = selectBalanced(scoringItems, scoringTask, { k: 3 })
    expect(selected.length).toBe(3)
  })
})

// ── Selection: diverse ──────────────────────────────────

describe('selectDiverse', () => {
  const task = { type: 'classification', labels: ['pos', 'neg'] }

  test('selects examples with diverse features', () => {
    const items = [
      { text: 'good great amazing', label: 'pos' },
      { text: 'good great wonderful', label: 'pos' },
      { text: 'bad terrible awful', label: 'neg' },
      { text: 'decent okay average', label: 'pos' },
    ]
    const selected = selectDiverse(items, task, { k: 3 })
    expect(selected.length).toBe(3)
    // Should prefer diverse labels
    const labels = new Set(selected.map(e => e.label))
    expect(labels.size).toBe(2) // both pos and neg
  })

  test('returns all if k >= length', () => {
    const items = [{ text: 'a', label: 'pos' }, { text: 'b', label: 'neg' }]
    const selected = selectDiverse(items, task, { k: 5 })
    expect(selected.length).toBe(2)
  })
})

// ── Selection: similar ──────────────────────────────────

describe('selectSimilar', () => {
  test('selects most similar to query', () => {
    const items = [
      { text: 'the cat sat on the mat' },
      { text: 'the dog ran in the park' },
      { text: 'the cat played with yarn' },
      { text: 'stocks rose sharply today' },
    ]
    const selected = selectSimilar(items, 'the cat slept on the mat', { k: 2 })
    expect(selected.length).toBe(2)
    // First result should be the most similar (cat + sat + mat)
    expect(selected[0].text).toBe('the cat sat on the mat')
  })
})

// ── selectExamples (unified interface) ──────────────────

describe('selectExamples', () => {
  const task = { type: 'classification', labels: ['pos', 'neg'] }
  const items = [
    { text: 'good', label: 'pos' },
    { text: 'bad', label: 'neg' },
    { text: 'great', label: 'pos' },
    { text: 'awful', label: 'neg' },
  ]

  test('routes to random', () => {
    const r = selectExamples('random', items, task, { k: 2 })
    expect(r.length).toBe(2)
  })

  test('routes to balanced', () => {
    const r = selectExamples('balanced', items, task, { k: 2 })
    expect(r.length).toBe(2)
  })

  test('routes to diverse', () => {
    const r = selectExamples('diverse', items, task, { k: 2 })
    expect(r.length).toBe(2)
  })

  test('routes to similar with query', () => {
    const r = selectExamples('similar', items, task, { k: 1, query: 'good stuff' })
    expect(r.length).toBe(1)
    expect(r[0].text).toBe('good')
  })

  test('throws for similar without query', () => {
    expect(() => selectExamples('similar', items, task, { k: 1 })).toThrow('query')
  })

  test('throws for unknown strategy', () => {
    expect(() => selectExamples('magic', items, task)).toThrow('Unknown strategy')
  })
})

// ── Prompt formatting ───────────────────────────────────

describe('formatExample', () => {
  test('formats classification example', () => {
    const out = formatExample({ text: 'great product', label: 'pos' }, { type: 'classification' })
    expect(out).toContain('Input: great product')
    expect(out).toContain('Label: pos')
  })

  test('formats scoring example', () => {
    const out = formatExample({ text: 'decent item', value: 3.5 }, { type: 'scoring' })
    expect(out).toContain('Input: decent item')
    expect(out).toContain('Score: 3.50')
  })

  test('formats sequence-labeling example', () => {
    const out = formatExample(
      { tokens: ['John', 'left'], tags: ['B-PER', 'O'] },
      { type: 'sequence-labeling' }
    )
    expect(out).toContain('Tokens:')
    expect(out).toContain('Tags:')
    expect(out).toContain('John')
  })

  test('formats extraction example', () => {
    const out = formatExample(
      { text: 'invoice 123', fields: { id: '123' } },
      { type: 'extraction' }
    )
    expect(out).toContain('Input: invoice 123')
    expect(out).toContain('"id"')
  })
})

describe('taskHeader', () => {
  test('classification header includes labels', () => {
    const h = taskHeader({ type: 'classification', description: 'Sentiment', labels: ['pos', 'neg'] })
    expect(h).toContain('Sentiment')
    expect(h).toContain('pos, neg')
  })

  test('scoring header includes range', () => {
    const h = taskHeader({ type: 'scoring', description: 'Rate', scoreRange: { min: 1, max: 10 } })
    expect(h).toContain('1')
    expect(h).toContain('10')
  })

  test('sequence-labeling header includes BIO', () => {
    const h = taskHeader({ type: 'sequence-labeling', description: 'NER', labels: ['PER', 'LOC'] })
    expect(h).toContain('BIO')
    expect(h).toContain('PER, LOC')
  })
})

describe('buildFewShotPrompt', () => {
  test('builds complete prompt with header, demos, query', () => {
    const task = { type: 'classification', description: 'Sentiment', labels: ['pos', 'neg'] }
    const examples = [
      { text: 'love it', label: 'pos' },
      { text: 'hate it', label: 'neg' },
    ]
    const prompt = buildFewShotPrompt(task, examples, 'it is okay')
    expect(prompt).toContain('Sentiment')
    expect(prompt).toContain('love it')
    expect(prompt).toContain('Label: pos')
    expect(prompt).toContain('hate it')
    expect(prompt).toContain('Label: neg')
    expect(prompt).toContain('it is okay')
  })

  test('handles sequence-labeling query', () => {
    const task = { type: 'sequence-labeling', description: 'NER', labels: ['PER'] }
    const examples = [{ tokens: ['John', 'left'], tags: ['B-PER', 'O'] }]
    const prompt = buildFewShotPrompt(task, examples, 'Alice ran')
    expect(prompt).toContain('Alice')
  })
})

// ── Persistence ─────────────────────────────────────────

describe('few-shot persistence', () => {
  const taskName = 'p17-persist-test'
  const taskModelsDir = join(ROOT, 'models', taskName)

  afterAll(async () => {
    await rm(taskModelsDir, { recursive: true, force: true }).catch(() => {})
  })

  test('loads null when no config exists', async () => {
    const config = await loadFewShotConfig('nonexistent-xyz-task')
    expect(config).toBeNull()
  })

  test('saves and loads config', async () => {
    const config = {
      strategy: 'diverse',
      k: 5,
      examples: [{ text: 'hello', label: 'pos' }],
      accuracy: 0.9,
      evaluatedAt: '2026-03-13T00:00:00Z'
    }
    await saveFewShotConfig(taskName, config)
    const loaded = await loadFewShotConfig(taskName)
    expect(loaded.strategy).toBe('diverse')
    expect(loaded.k).toBe(5)
    expect(loaded.accuracy).toBe(0.9)
    expect(loaded.examples.length).toBe(1)
  })
})

// ── Exports ─────────────────────────────────────────────

describe('exports', () => {
  test('all expected functions are exported', () => {
    const fns = [
      tokenize, ngrams, jaccard, textSimilarity,
      exampleFeatures,
      selectRandom, selectBalanced, selectDiverse, selectSimilar,
      selectExamples,
      formatExample, buildFewShotPrompt, taskHeader,
      evaluatePromptSet, optimizeFewShot,
      saveFewShotConfig, loadFewShotConfig
    ]
    for (const fn of fns) {
      expect(typeof fn).toBe('function')
    }
  })
})
