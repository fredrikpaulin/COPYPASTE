import { test, expect, describe } from 'bun:test'

// ── lib/curriculum.js — pure functions ────────────────────

describe('sortByCurriculum', () => {
  test('sorts by difficulty ascending', async () => {
    const { sortByCurriculum } = await import('../lib/curriculum.js')
    const data = [
      { text: 'hard', _difficulty: 0.9 },
      { text: 'easy', _difficulty: 0.1 },
      { text: 'medium', _difficulty: 0.5 }
    ]
    const sorted = sortByCurriculum(data)
    expect(sorted[0].text).toBe('easy')
    expect(sorted[1].text).toBe('medium')
    expect(sorted[2].text).toBe('hard')
  })

  test('handles missing _difficulty as 0', async () => {
    const { sortByCurriculum } = await import('../lib/curriculum.js')
    const data = [
      { text: 'scored', _difficulty: 0.5 },
      { text: 'unscored' }
    ]
    const sorted = sortByCurriculum(data)
    expect(sorted[0].text).toBe('unscored')
    expect(sorted[1].text).toBe('scored')
  })

  test('does not mutate original array', async () => {
    const { sortByCurriculum } = await import('../lib/curriculum.js')
    const data = [{ _difficulty: 0.9 }, { _difficulty: 0.1 }]
    const sorted = sortByCurriculum(data)
    expect(data[0]._difficulty).toBe(0.9) // unchanged
    expect(sorted[0]._difficulty).toBe(0.1)
  })
})

describe('curriculumStages', () => {
  test('splits into easy, medium, hard', async () => {
    const { curriculumStages } = await import('../lib/curriculum.js')
    const data = [
      { text: 'a', _difficulty: 0.1 },
      { text: 'b', _difficulty: 0.2 },
      { text: 'c', _difficulty: 0.5 },
      { text: 'd', _difficulty: 0.6 },
      { text: 'e', _difficulty: 0.8 },
      { text: 'f', _difficulty: 0.95 }
    ]
    const { easy, medium, hard } = curriculumStages(data)
    expect(easy.length).toBe(2)
    expect(medium.length).toBe(2)
    expect(hard.length).toBe(2)
  })

  test('respects custom thresholds', async () => {
    const { curriculumStages } = await import('../lib/curriculum.js')
    const data = [
      { _difficulty: 0.1 },
      { _difficulty: 0.4 },
      { _difficulty: 0.6 },
      { _difficulty: 0.9 }
    ]
    const { easy, medium, hard } = curriculumStages(data, { easyThreshold: 0.5, hardThreshold: 0.8 })
    expect(easy.length).toBe(2)  // 0.1, 0.4
    expect(medium.length).toBe(1) // 0.6
    expect(hard.length).toBe(1)  // 0.9
  })

  test('handles empty data', async () => {
    const { curriculumStages } = await import('../lib/curriculum.js')
    const { easy, medium, hard } = curriculumStages([])
    expect(easy.length).toBe(0)
    expect(medium.length).toBe(0)
    expect(hard.length).toBe(0)
  })
})

describe('filterByQuality', () => {
  test('filters below threshold', async () => {
    const { filterByQuality } = await import('../lib/curriculum.js')
    const data = [
      { text: 'good', _quality: 0.9 },
      { text: 'ok', _quality: 0.7 },
      { text: 'bad', _quality: 0.3 },
      { text: 'terrible', _quality: 0.1 }
    ]
    const result = filterByQuality(data, 0.7)
    expect(result.data.length).toBe(2)
    expect(result.removed).toBe(2)
  })

  test('keeps all when threshold is 0', async () => {
    const { filterByQuality } = await import('../lib/curriculum.js')
    const data = [{ _quality: 0.1 }, { _quality: 0.01 }]
    const result = filterByQuality(data, 0)
    expect(result.data.length).toBe(2)
    expect(result.removed).toBe(0)
  })

  test('treats missing _quality as 0', async () => {
    const { filterByQuality } = await import('../lib/curriculum.js')
    const data = [{ text: 'no score' }, { text: 'scored', _quality: 0.8 }]
    const result = filterByQuality(data, 0.5)
    expect(result.data.length).toBe(1)
    expect(result.data[0].text).toBe('scored')
  })
})

// ── generateContrastive validation ─────────────────────────

describe('generateContrastive — input validation', () => {
  test('throws for non-classification task', async () => {
    const { generateContrastive } = await import('../lib/curriculum.js')
    try {
      await generateContrastive({ type: 'extraction', labels: ['a'] }, {})
      expect(true).toBe(false) // should not reach
    } catch (e) {
      expect(e.message).toContain('classification task with 2+ labels')
    }
  })

  test('throws for single-label classification', async () => {
    const { generateContrastive } = await import('../lib/curriculum.js')
    try {
      await generateContrastive({ type: 'classification', labels: ['only'] }, {})
      expect(true).toBe(false)
    } catch (e) {
      expect(e.message).toContain('2+ labels')
    }
  })
})

// ── llmJudge with mock server ───────────────────────────────

describe('llmJudge — mock provider', () => {
  let server
  let requestCount = 0

  // We mock the provider module by creating a local HTTP server
  // and using callProvider indirectly. Instead, test the scoring logic
  // by validating the exported functions work with pre-scored data.

  test('scored data flows through filterByQuality', async () => {
    const { filterByQuality } = await import('../lib/curriculum.js')
    // Simulate what llmJudge returns
    const scored = [
      { text: 'good example', label: 'pos', _quality: 0.9, _relevance: 0.9, _naturalness: 0.9, _label_correctness: 0.9 },
      { text: 'bad example', label: 'neg', _quality: 0.3, _relevance: 0.2, _naturalness: 0.3, _label_correctness: 0.4 },
      { text: 'ok example', label: 'pos', _quality: 0.75, _relevance: 0.7, _naturalness: 0.8, _label_correctness: 0.75 }
    ]
    const result = filterByQuality(scored, 0.7)
    expect(result.data.length).toBe(2)
    expect(result.data.every(d => d._quality >= 0.7)).toBe(true)
  })
})
