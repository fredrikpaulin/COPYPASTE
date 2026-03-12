import { test, expect, describe } from 'bun:test'
import {
  deduplicate, trigrams, trigramSimilarity,
  augment, synonymReplace, randomInsert,
  labelCounts, labelImbalance
} from '../lib/data.js'

// ── Trigrams ──────────────────────────────────────────────

describe('trigrams', () => {
  test('generates correct trigrams', () => {
    const t = trigrams('hello')
    expect(t.has('hel')).toBe(true)
    expect(t.has('ell')).toBe(true)
    expect(t.has('llo')).toBe(true)
  })

  test('lowercases input', () => {
    const t = trigrams('ABC')
    expect(t.has('abc')).toBe(true)
  })

  test('returns empty set for short strings', () => {
    expect(trigrams('ab').size).toBe(0)
    expect(trigrams('a').size).toBe(0)
  })
})

describe('trigramSimilarity', () => {
  test('identical strings return 1.0', () => {
    const a = trigrams('hello world')
    expect(trigramSimilarity(a, a)).toBe(1)
  })

  test('completely different strings return ~0', () => {
    const a = trigrams('aaaaaaa')
    const b = trigrams('zzzzzzz')
    expect(trigramSimilarity(a, b)).toBeLessThan(0.1)
  })

  test('similar strings score high', () => {
    const a = trigrams('great product love it')
    const b = trigrams('great product loved it')
    expect(trigramSimilarity(a, b)).toBeGreaterThan(0.7)
  })
})

// ── Deduplication ─────────────────────────────────────────

describe('deduplicate', () => {
  test('removes exact duplicates', () => {
    const data = [
      { text: 'hello', label: 'a' },
      { text: 'hello', label: 'a' },
      { text: 'world', label: 'b' }
    ]
    const result = deduplicate(data)
    expect(result.data).toHaveLength(2)
    expect(result.removed).toBe(1)
  })

  test('removes fuzzy duplicates above threshold', () => {
    const data = [
      { text: 'This is a great product', label: 'pos' },
      { text: 'This is a great products', label: 'pos' }, // very similar
      { text: 'Completely different text here', label: 'neg' }
    ]
    const result = deduplicate(data, { fuzzyThreshold: 0.8 })
    expect(result.data).toHaveLength(2)
    expect(result.removed).toBe(1)
  })

  test('keeps dissimilar items', () => {
    const data = [
      { text: 'The cat sat on the mat', label: 'a' },
      { text: 'A dog ran through the park', label: 'b' },
      { text: 'Fish swim in the ocean', label: 'c' }
    ]
    const result = deduplicate(data)
    expect(result.data).toHaveLength(3)
    expect(result.removed).toBe(0)
  })

  test('handles empty data', () => {
    const result = deduplicate([])
    expect(result.data).toHaveLength(0)
    expect(result.removed).toBe(0)
  })

  test('disabling fuzzy with threshold 1 only removes exact', () => {
    const data = [
      { text: 'hello world', label: 'a' },
      { text: 'hello worlds', label: 'a' } // almost identical but not exact
    ]
    const result = deduplicate(data, { fuzzyThreshold: 1 })
    expect(result.data).toHaveLength(2)
  })
})

// ── Augmentation ──────────────────────────────────────────

describe('synonymReplace', () => {
  test('returns a string', () => {
    const result = synonymReplace('This is a good product')
    expect(typeof result).toBe('string')
  })

  test('with probability 0 returns original', () => {
    const text = 'This is a good product'
    const result = synonymReplace(text, 0)
    expect(result).toBe(text)
  })

  test('preserves word count approximately', () => {
    const text = 'I like this very good product'
    const result = synonymReplace(text, 1)
    // Might change words but shouldn't wildly change count
    const origCount = text.split(/\s+/).length
    const newCount = result.split(/\s+/).length
    expect(newCount).toBe(origCount)
  })
})

describe('randomInsert', () => {
  test('returns a string', () => {
    const result = randomInsert('hello world')
    expect(typeof result).toBe('string')
  })

  test('with probability 0 returns original', () => {
    const text = 'hello world'
    expect(randomInsert(text, 0)).toBe(text)
  })

  test('with high probability adds words', () => {
    const text = 'hello world foo bar baz'
    // Run multiple times, at least once it should be longer
    let longerOnce = false
    for (let i = 0; i < 20; i++) {
      const result = randomInsert(text, 0.9)
      if (result.split(/\s+/).length > text.split(/\s+/).length) {
        longerOnce = true
        break
      }
    }
    expect(longerOnce).toBe(true)
  })
})

describe('augment', () => {
  test('increases dataset size', () => {
    const data = [
      { text: 'I like this good product very much', label: 'pos' },
      { text: 'This is bad and slow terrible', label: 'neg' }
    ]
    const result = augment(data, { multiplier: 3 })
    expect(result.length).toBeGreaterThan(data.length)
  })

  test('preserves original data', () => {
    const data = [{ text: 'original text', label: 'a' }]
    const result = augment(data, { multiplier: 2 })
    expect(result[0].text).toBe('original text')
    expect(result[0].label).toBe('a')
  })

  test('marks augmented examples', () => {
    const data = [{ text: 'I really like this very good nice product', label: 'pos' }]
    const result = augment(data, { multiplier: 5, synonymProb: 1, insertProb: 1 })
    const augmented = result.filter(r => r._augmented)
    expect(augmented.length).toBeGreaterThan(0)
  })

  test('multiplier 1 returns only originals', () => {
    const data = [{ text: 'test', label: 'a' }]
    const result = augment(data, { multiplier: 1 })
    expect(result).toHaveLength(1)
  })
})

// ── Label Balance ─────────────────────────────────────────

describe('labelCounts', () => {
  test('counts labels correctly', () => {
    const data = [
      { label: 'a' }, { label: 'b' }, { label: 'a' }, { label: 'c' }, { label: 'a' }
    ]
    const counts = labelCounts(data)
    expect(counts.a).toBe(3)
    expect(counts.b).toBe(1)
    expect(counts.c).toBe(1)
  })

  test('handles empty data', () => {
    expect(labelCounts([])).toEqual({})
  })
})

describe('labelImbalance', () => {
  test('detects under-represented labels', () => {
    const data = [
      ...Array(80).fill({ label: 'pos' }),
      ...Array(10).fill({ label: 'neg' }),
      ...Array(10).fill({ label: 'neutral' })
    ]
    const deficit = labelImbalance(data, ['pos', 'neg', 'neutral'])
    // Expected ~33 each, neg and neutral are at 10 which is well below 80% of 33
    expect(deficit.neg).toBeGreaterThan(0)
    expect(deficit.neutral).toBeGreaterThan(0)
    expect(deficit.pos).toBeUndefined()
  })

  test('returns empty when balanced', () => {
    const data = [
      ...Array(33).fill({ label: 'a' }),
      ...Array(33).fill({ label: 'b' }),
      ...Array(34).fill({ label: 'c' })
    ]
    const deficit = labelImbalance(data, ['a', 'b', 'c'])
    expect(Object.keys(deficit)).toHaveLength(0)
  })

  test('handles missing labels', () => {
    const data = Array(100).fill({ label: 'a' })
    const deficit = labelImbalance(data, ['a', 'b'])
    expect(deficit.b).toBeGreaterThan(0)
  })
})
