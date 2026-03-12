import { test, expect, describe } from 'bun:test'

// ── kFoldSplit ──────────────────────────────────────────────

describe('kFoldSplit', () => {
  test('returns k folds', async () => {
    const { kFoldSplit } = await import('../lib/evaluate.js')
    const folds = kFoldSplit(100, 5)
    expect(folds.length).toBe(5)
  })

  test('each fold has train and val', async () => {
    const { kFoldSplit } = await import('../lib/evaluate.js')
    const folds = kFoldSplit(50, 5)
    for (const fold of folds) {
      expect(fold).toHaveProperty('train')
      expect(fold).toHaveProperty('val')
      expect(fold.train.length + fold.val.length).toBe(50)
    }
  })

  test('val sets are disjoint', async () => {
    const { kFoldSplit } = await import('../lib/evaluate.js')
    const folds = kFoldSplit(20, 4)
    const allVal = new Set()
    for (const fold of folds) {
      for (const idx of fold.val) {
        expect(allVal.has(idx)).toBe(false)
        allVal.add(idx)
      }
    }
    expect(allVal.size).toBe(20)
  })

  test('all indices appear exactly once in val across folds', async () => {
    const { kFoldSplit } = await import('../lib/evaluate.js')
    const folds = kFoldSplit(15, 3)
    const valCounts = new Map()
    for (const fold of folds) {
      for (const idx of fold.val) {
        valCounts.set(idx, (valCounts.get(idx) || 0) + 1)
      }
    }
    for (const count of valCounts.values()) {
      expect(count).toBe(1)
    }
  })

  test('default k is 5', async () => {
    const { kFoldSplit } = await import('../lib/evaluate.js')
    const folds = kFoldSplit(25)
    expect(folds.length).toBe(5)
  })
})

// ── errorTaxonomy ───────────────────────────────────────────

describe('errorTaxonomy', () => {
  test('counts correct and errors', async () => {
    const { errorTaxonomy } = await import('../lib/evaluate.js')
    const valData = [
      { text: 'a', label: 'pos' },
      { text: 'b', label: 'neg' },
      { text: 'c', label: 'pos' },
      { text: 'd', label: 'neg' }
    ]
    const predictions = ['pos', 'pos', 'neg', 'neg']

    const result = errorTaxonomy(valData, predictions)
    expect(result.totalCorrect).toBe(2) // a=pos, d=neg
    expect(result.totalErrors).toBe(2)  // b=neg→pos, c=pos→neg
  })

  test('tracks confusion pairs', async () => {
    const { errorTaxonomy } = await import('../lib/evaluate.js')
    const valData = [
      { text: 'x', label: 'A' },
      { text: 'y', label: 'A' },
      { text: 'z', label: 'B' }
    ]
    const predictions = ['B', 'B', 'A']

    const result = errorTaxonomy(valData, predictions)
    expect(result.confusionPairs['A → B']).toBe(2)
    expect(result.confusionPairs['B → A']).toBe(1)
    expect(result.topConfusions[0].pair).toBe('A → B')
  })

  test('categorizes errors by text length', async () => {
    const { errorTaxonomy } = await import('../lib/evaluate.js')
    const valData = [
      { text: 'short', label: 'A' },                                    // <50 chars
      { text: 'a medium length text that is between fifty and one hundred fifty characters long yes it is', label: 'A' },  // 50-150
      { text: 'x'.repeat(200), label: 'A' }                             // >150
    ]
    const predictions = ['B', 'B', 'B']

    const result = errorTaxonomy(valData, predictions)
    expect(result.byTextLength.short).toBe(1)
    expect(result.byTextLength.medium).toBe(1)
    expect(result.byTextLength.long).toBe(1)
  })

  test('handles prediction objects with .label', async () => {
    const { errorTaxonomy } = await import('../lib/evaluate.js')
    const valData = [{ text: 'x', label: 'A' }]
    const predictions = [{ label: 'B', confidence: 0.6 }]

    const result = errorTaxonomy(valData, predictions)
    expect(result.totalErrors).toBe(1)
    expect(result.confusionPairs['A → B']).toBe(1)
  })

  test('tracks errors by provider', async () => {
    const { errorTaxonomy } = await import('../lib/evaluate.js')
    const valData = [
      { text: 'a', label: 'X', _provider: 'openai' },
      { text: 'b', label: 'X', _provider: 'anthropic' },
      { text: 'c', label: 'X', _source: 'ollama' }
    ]
    const predictions = ['Y', 'Y', 'Y']

    const result = errorTaxonomy(valData, predictions)
    expect(result.byProvider['openai']).toBe(1)
    expect(result.byProvider['anthropic']).toBe(1)
    expect(result.byProvider['ollama']).toBe(1)
  })

  test('handles all correct predictions', async () => {
    const { errorTaxonomy } = await import('../lib/evaluate.js')
    const valData = [
      { text: 'a', label: 'pos' },
      { text: 'b', label: 'neg' }
    ]
    const predictions = ['pos', 'neg']

    const result = errorTaxonomy(valData, predictions)
    expect(result.totalCorrect).toBe(2)
    expect(result.totalErrors).toBe(0)
    expect(result.topConfusions.length).toBe(0)
  })
})

// ── calibrationBins ─────────────────────────────────────────

describe('calibrationBins', () => {
  test('returns bins and ECE', async () => {
    const { calibrationBins } = await import('../lib/evaluate.js')
    const predictions = [
      { label: 'A', confidence: 0.9 },
      { label: 'B', confidence: 0.8 },
      { label: 'A', confidence: 0.6 },
      { label: 'B', confidence: 0.3 }
    ]
    const actual = ['A', 'B', 'B', 'B']

    const result = calibrationBins(predictions, actual)
    expect(result.bins.length).toBe(10)
    expect(typeof result.ece).toBe('number')
    expect(result.ece).toBeGreaterThanOrEqual(0)
    expect(result.hasConfidence).toBe(true)
    expect(result.total).toBe(4)
  })

  test('returns hasConfidence=false when no confidence scores', async () => {
    const { calibrationBins } = await import('../lib/evaluate.js')
    const predictions = [{ label: 'A' }, { label: 'B' }]
    const actual = ['A', 'B']

    const result = calibrationBins(predictions, actual)
    expect(result.hasConfidence).toBe(false)
    expect(result.bins.length).toBe(0)
  })

  test('perfect calibration has low ECE', async () => {
    const { calibrationBins } = await import('../lib/evaluate.js')
    // 10 predictions all at 0.95 confidence, 9 correct = ~0.90 accuracy in that bin
    const predictions = Array.from({ length: 10 }, (_, i) => ({
      label: 'A',
      confidence: 0.95
    }))
    const actual = Array.from({ length: 10 }, (_, i) => i < 9 ? 'A' : 'B')

    const result = calibrationBins(predictions, actual)
    // ECE should be |0.9 - 0.95| = 0.05
    expect(result.ece).toBeLessThan(0.1)
  })

  test('respects nBins parameter', async () => {
    const { calibrationBins } = await import('../lib/evaluate.js')
    const predictions = [{ label: 'A', confidence: 0.5 }]
    const actual = ['A']

    const result = calibrationBins(predictions, actual, { nBins: 5 })
    expect(result.bins.length).toBe(5)
  })
})

// ── projectTo2D ─────────────────────────────────────────────

describe('projectTo2D', () => {
  test('returns 2D coordinates', async () => {
    const { projectTo2D } = await import('../lib/evaluate.js')
    const embeddings = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      [1, 1, 0, 0]
    ]
    const result = projectTo2D(embeddings)
    expect(result.length).toBe(5)
    for (const point of result) {
      expect(point).toHaveProperty('x')
      expect(point).toHaveProperty('y')
      expect(typeof point.x).toBe('number')
      expect(typeof point.y).toBe('number')
    }
  })

  test('returns empty for empty input', async () => {
    const { projectTo2D } = await import('../lib/evaluate.js')
    expect(projectTo2D([])).toEqual([])
  })

  test('separates distinct clusters', async () => {
    const { projectTo2D } = await import('../lib/evaluate.js')
    // Two clear clusters
    const embeddings = [
      [10, 10, 0], [10.1, 10.1, 0], [9.9, 9.9, 0],
      [-10, -10, 0], [-10.1, -10.1, 0], [-9.9, -9.9, 0]
    ]
    const result = projectTo2D(embeddings)

    // Points 0-2 should be close together, points 3-5 close together
    // and the two groups should be far apart
    const dist = (a, b) => Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
    const intraCluster = dist(result[0], result[1])
    const interCluster = dist(result[0], result[3])
    expect(interCluster).toBeGreaterThan(intraCluster * 5)
  })

  test('projects to 2D even with high-dimensional input', async () => {
    const { projectTo2D } = await import('../lib/evaluate.js')
    // 50-dimensional input
    const embeddings = Array.from({ length: 10 }, (_, i) => {
      const v = new Array(50).fill(0)
      v[i % 50] = 1
      return v
    })
    const result = projectTo2D(embeddings)
    expect(result.length).toBe(10)
    // All points should have finite coordinates
    for (const p of result) {
      expect(Number.isFinite(p.x)).toBe(true)
      expect(Number.isFinite(p.y)).toBe(true)
    }
  })
})

// ── featureImportance — export check ─────────────────────────

describe('featureImportance', () => {
  test('exported function exists', async () => {
    const { featureImportance } = await import('../lib/evaluate.js')
    expect(typeof featureImportance).toBe('function')
  })
})

// ── kFoldCV — export check ───────────────────────────────────

describe('kFoldCV', () => {
  test('exported function exists', async () => {
    const { kFoldCV } = await import('../lib/evaluate.js')
    expect(typeof kFoldCV).toBe('function')
  })
})
