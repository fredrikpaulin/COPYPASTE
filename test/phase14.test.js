import { test, expect, describe, afterAll } from 'bun:test'
import {
  extractFeatures, featureHash, fnv1a, wordShape,
  viterbi, score,
  trainCRF, predictSequence, predictBatch,
  extractEntities, evaluateEntities,
  saveModel, loadModel, hasCRFModel,
  labelsToBIO, validateBIO
} from '../lib/crf.js'
import { buildSystemPrompt, buildBatchPrompt, validateExample } from '../lib/generate.js'
import { loadTemplate, listTemplates } from '../lib/templates.js'
import { rm } from 'node:fs/promises'
import { join } from 'node:path'

const MODELS_DIR = join(import.meta.dir, '..', 'models')

afterAll(async () => {
  await rm(join(MODELS_DIR, 'test-crf'), { recursive: true, force: true }).catch(() => {})
})

// ── Feature extraction ───────────────────────────────────

describe('wordShape', () => {
  test('maps uppercase to X, lowercase to x, digits to d', () => {
    expect(wordShape('Hello')).toBe('Xx')
    expect(wordShape('USA')).toBe('X')
    expect(wordShape('hello123')).toBe('xd')
  })

  test('collapses repeated shapes', () => {
    expect(wordShape('AAA')).toBe('X')
    expect(wordShape('abc')).toBe('x')
  })
})

describe('extractFeatures', () => {
  const tokens = ['John', 'lives', 'in', 'New', 'York']

  test('includes word, shape, prefix, suffix features', () => {
    const feats = extractFeatures(tokens, 0, '<S>')
    expect(feats).toContain('w=john')
    expect(feats).toContain('cap=Y')
    expect(feats.some(f => f.startsWith('shape='))).toBe(true)
    expect(feats.some(f => f.startsWith('pre2='))).toBe(true)
    expect(feats.some(f => f.startsWith('suf2='))).toBe(true)
  })

  test('includes BOS for first position', () => {
    const feats = extractFeatures(tokens, 0, '<S>')
    expect(feats).toContain('BOS')
  })

  test('includes EOS for last position', () => {
    const feats = extractFeatures(tokens, 4, 'O')
    expect(feats).toContain('EOS')
  })

  test('includes prev word bigram', () => {
    const feats = extractFeatures(tokens, 1, 'B-PER')
    expect(feats).toContain('pw=john')
    expect(feats).toContain('bigram=john|lives')
    expect(feats).toContain('prev_tag=B-PER')
  })

  test('includes next word', () => {
    const feats = extractFeatures(tokens, 1, 'O')
    expect(feats).toContain('nw=in')
  })
})

// ── Feature hashing ──────────────────────────────────────

describe('fnv1a', () => {
  test('produces consistent hashes', () => {
    expect(fnv1a('hello')).toBe(fnv1a('hello'))
    expect(fnv1a('hello')).not.toBe(fnv1a('world'))
  })

  test('returns a positive integer', () => {
    expect(fnv1a('test')).toBeGreaterThan(0)
    expect(Number.isInteger(fnv1a('test'))).toBe(true)
  })
})

describe('featureHash', () => {
  test('maps feature+tag to bounded index', () => {
    const h = featureHash('w=hello', 'O', 1024)
    expect(h).toBeGreaterThanOrEqual(0)
    expect(h).toBeLessThan(1024)
  })
})

// ── BIO utilities ────────────────────────────────────────

describe('labelsToBIO', () => {
  test('converts entity types to BIO tag set', () => {
    const tags = labelsToBIO(['PER', 'LOC'])
    expect(tags).toContain('O')
    expect(tags).toContain('B-PER')
    expect(tags).toContain('I-PER')
    expect(tags).toContain('B-LOC')
    expect(tags).toContain('I-LOC')
    expect(tags.length).toBe(5)
  })
})

describe('validateBIO', () => {
  test('accepts valid BIO sequence', () => {
    const errors = validateBIO(['B-PER', 'I-PER', 'O', 'B-LOC'])
    expect(errors.length).toBe(0)
  })

  test('catches I- without preceding B-', () => {
    const errors = validateBIO(['O', 'I-PER', 'O'])
    expect(errors.length).toBe(1)
    expect(errors[0].position).toBe(1)
  })

  test('catches I- with wrong type', () => {
    const errors = validateBIO(['B-PER', 'I-LOC', 'O'])
    expect(errors.length).toBe(1)
    expect(errors[0].position).toBe(1)
  })
})

// ── Entity extraction ────────────────────────────────────

describe('extractEntities', () => {
  test('extracts single-token entities', () => {
    const entities = extractEntities(
      ['John', 'went', 'home'],
      ['B-PER', 'O', 'O']
    )
    expect(entities.length).toBe(1)
    expect(entities[0]).toEqual({ type: 'PER', start: 0, end: 1, tokens: ['John'], text: 'John' })
  })

  test('extracts multi-token entities', () => {
    const entities = extractEntities(
      ['New', 'York', 'City'],
      ['B-LOC', 'I-LOC', 'I-LOC']
    )
    expect(entities.length).toBe(1)
    expect(entities[0].text).toBe('New York City')
    expect(entities[0].start).toBe(0)
    expect(entities[0].end).toBe(3)
  })

  test('extracts multiple entities', () => {
    const entities = extractEntities(
      ['John', 'visited', 'Paris', 'and', 'London'],
      ['B-PER', 'O', 'B-LOC', 'O', 'B-LOC']
    )
    expect(entities.length).toBe(3)
    expect(entities[0].type).toBe('PER')
    expect(entities[1].type).toBe('LOC')
    expect(entities[2].type).toBe('LOC')
  })

  test('returns empty for all-O sequence', () => {
    const entities = extractEntities(['the', 'cat', 'sat'], ['O', 'O', 'O'])
    expect(entities.length).toBe(0)
  })
})

// ── Viterbi decoding ─────────────────────────────────────

describe('viterbi', () => {
  test('returns path of correct length', () => {
    const tags = ['O', 'B-PER', 'I-PER']
    const hashSize = 1024
    const weights = new Float64Array(hashSize)
    const { path } = viterbi(['John', 'lives'], tags, weights, hashSize)
    expect(path.length).toBe(2)
    expect(tags).toContain(path[0])
    expect(tags).toContain(path[1])
  })

  test('returns empty path for empty tokens', () => {
    const { path } = viterbi([], ['O'], new Float64Array(1024), 1024)
    expect(path.length).toBe(0)
  })

  test('responds to weight changes', () => {
    const tags = ['O', 'B-PER']
    const hashSize = 1024
    const weights = new Float64Array(hashSize)

    // Heavily weight B-PER features for "John"
    const feats = extractFeatures(['John'], 0, '<S>')
    for (const f of feats) {
      weights[featureHash(f, 'B-PER', hashSize)] = 10
      weights[featureHash(f, 'O', hashSize)] = -10
    }

    const { path } = viterbi(['John'], tags, weights, hashSize)
    expect(path[0]).toBe('B-PER')
  })
})

// ── Training ─────────────────────────────────────────────

describe('trainCRF', () => {
  const trainingData = [
    { tokens: ['John', 'lives', 'in', 'London'], tags: ['B-PER', 'O', 'O', 'B-LOC'] },
    { tokens: ['Mary', 'works', 'at', 'Google'], tags: ['B-PER', 'O', 'O', 'B-ORG'] },
    { tokens: ['The', 'cat', 'sat'], tags: ['O', 'O', 'O'] },
    { tokens: ['Bob', 'visited', 'Paris'], tags: ['B-PER', 'O', 'B-LOC'] },
    { tokens: ['Alice', 'joined', 'Microsoft'], tags: ['B-PER', 'O', 'B-ORG'] },
    { tokens: ['It', 'was', 'sunny'], tags: ['O', 'O', 'O'] },
    { tokens: ['Tim', 'flew', 'to', 'Tokyo'], tags: ['B-PER', 'O', 'O', 'B-LOC'] },
    { tokens: ['Eve', 'left', 'Apple'], tags: ['B-PER', 'O', 'B-ORG'] },
  ]
  const tags = labelsToBIO(['PER', 'LOC', 'ORG'])

  test('returns model with weights, tags, hashSize', () => {
    const model = trainCRF(trainingData, { tags, epochs: 3 })
    expect(model.weights).toBeInstanceOf(Float64Array)
    expect(model.tags).toEqual(tags)
    expect(model.hashSize).toBeGreaterThan(0)
  })

  test('calls onEpoch callback', () => {
    const calls = []
    trainCRF(trainingData, {
      tags,
      epochs: 3,
      onEpoch: (info) => calls.push(info)
    })
    expect(calls.length).toBe(3)
    expect(calls[0].epoch).toBe(1)
    expect(calls[2].epoch).toBe(3)
    expect(calls[0].sequencesTotal).toBe(trainingData.length)
  })

  test('accuracy improves over epochs', () => {
    const accuracies = []
    trainCRF(trainingData, {
      tags,
      epochs: 15,
      onEpoch: ({ accuracy }) => accuracies.push(accuracy)
    })
    // Later epochs should generally be better (or at least not all zero)
    const lastThree = accuracies.slice(-3)
    const firstThree = accuracies.slice(0, 3)
    const avgLast = lastThree.reduce((a, b) => a + b) / lastThree.length
    const avgFirst = firstThree.reduce((a, b) => a + b) / firstThree.length
    expect(avgLast).toBeGreaterThanOrEqual(avgFirst)
  })
})

// ── Prediction ───────────────────────────────────────────

describe('predictSequence', () => {
  const trainingData = [
    { tokens: ['John', 'lives', 'in', 'London'], tags: ['B-PER', 'O', 'O', 'B-LOC'] },
    { tokens: ['Mary', 'works', 'at', 'Google'], tags: ['B-PER', 'O', 'O', 'B-ORG'] },
    { tokens: ['Bob', 'visited', 'Paris'], tags: ['B-PER', 'O', 'B-LOC'] },
    { tokens: ['Alice', 'joined', 'Microsoft'], tags: ['B-PER', 'O', 'B-ORG'] },
    { tokens: ['Tim', 'flew', 'to', 'Tokyo'], tags: ['B-PER', 'O', 'O', 'B-LOC'] },
    { tokens: ['Eve', 'left', 'Apple'], tags: ['B-PER', 'O', 'B-ORG'] },
    { tokens: ['The', 'cat', 'sat'], tags: ['O', 'O', 'O'] },
    { tokens: ['It', 'was', 'sunny'], tags: ['O', 'O', 'O'] },
  ]
  const tags = labelsToBIO(['PER', 'LOC', 'ORG'])

  test('returns tag array of correct length', () => {
    const model = trainCRF(trainingData, { tags, epochs: 10 })
    const result = predictSequence(['John', 'lives', 'in', 'London'], model)
    expect(result.length).toBe(4)
    for (const t of result) expect(tags).toContain(t)
  })

  test('predictBatch returns multiple results', () => {
    const model = trainCRF(trainingData, { tags, epochs: 10 })
    const results = predictBatch([
      ['John', 'lives'],
      ['The', 'cat']
    ], model)
    expect(results.length).toBe(2)
    expect(results[0].tokens).toEqual(['John', 'lives'])
    expect(results[0].tags.length).toBe(2)
  })
})

// ── Evaluation ───────────────────────────────────────────

describe('evaluateEntities', () => {
  test('computes entity-level metrics', () => {
    const gold = [
      { tokens: ['John', 'went', 'to', 'Paris'], tags: ['B-PER', 'O', 'O', 'B-LOC'] }
    ]
    const pred = [
      { tokens: ['John', 'went', 'to', 'Paris'], tags: ['B-PER', 'O', 'O', 'B-LOC'] }
    ]
    const eval_ = evaluateEntities(gold, pred)
    expect(eval_.micro.f1).toBe(1.0)
    expect(eval_.tokenAccuracy).toBe(1.0)
  })

  test('handles partial matches', () => {
    const gold = [
      { tokens: ['New', 'York', 'City'], tags: ['B-LOC', 'I-LOC', 'I-LOC'] }
    ]
    const pred = [
      { tokens: ['New', 'York', 'City'], tags: ['B-LOC', 'I-LOC', 'O'] }
    ]
    const eval_ = evaluateEntities(gold, pred)
    // Gold entity is "New York City" (0:3), pred is "New York" (0:2) — different span, so no match
    expect(eval_.micro.f1).toBe(0)
    expect(eval_.tokenAccuracy).toBeCloseTo(2 / 3, 2)
  })

  test('returns per-type metrics', () => {
    const gold = [
      { tokens: ['John', 'visited', 'Paris'], tags: ['B-PER', 'O', 'B-LOC'] }
    ]
    const pred = [
      { tokens: ['John', 'visited', 'Paris'], tags: ['B-PER', 'O', 'O'] }
    ]
    const eval_ = evaluateEntities(gold, pred)
    expect(eval_.byType.PER.f1).toBe(1.0) // PER correct
    expect(eval_.byType.LOC.recall).toBe(0) // LOC missed
  })
})

// ── Model persistence ────────────────────────────────────

describe('model save/load', () => {
  const tags = labelsToBIO(['PER', 'LOC'])
  const data = [
    { tokens: ['John', 'lives', 'in', 'London'], tags: ['B-PER', 'O', 'O', 'B-LOC'] },
    { tokens: ['The', 'cat'], tags: ['O', 'O'] },
  ]

  test('save and load roundtrip', async () => {
    const model = trainCRF(data, { tags, epochs: 3, hashSize: 1024 })
    await saveModel('test-crf', model)

    const loaded = await loadModel('test-crf')
    expect(loaded).not.toBeNull()
    expect(loaded.tags).toEqual(tags)
    expect(loaded.hashSize).toBe(1024)
    expect(loaded.weights.length).toBe(1024)
  })

  test('hasCRFModel returns true after save', async () => {
    expect(await hasCRFModel('test-crf')).toBe(true)
  })

  test('hasCRFModel returns false for nonexistent', async () => {
    expect(await hasCRFModel('nonexistent-crf-task')).toBe(false)
  })

  test('loaded model produces same predictions', async () => {
    const model = trainCRF(data, { tags, epochs: 5, hashSize: 1024 })
    await saveModel('test-crf', model)
    const loaded = await loadModel('test-crf')

    const input = ['John', 'lives', 'in', 'London']
    const orig = predictSequence(input, model)
    const fromLoaded = predictSequence(input, loaded)
    expect(fromLoaded).toEqual(orig)
  })
})

// ── Generate.js integration ──────────────────────────────

describe('generate.js sequence-labeling support', () => {
  const seqTask = {
    name: 'test-ner',
    type: 'sequence-labeling',
    description: 'NER task',
    labels: ['PER', 'LOC', 'ORG'],
    synthetic: { count: 10, prompt: 'Generate NER examples', batchSize: 5 }
  }

  test('buildSystemPrompt includes entity types', () => {
    const prompt = buildSystemPrompt(seqTask)
    expect(prompt).toContain('PER')
    expect(prompt).toContain('BIO')
  })

  test('buildBatchPrompt uses tokens/tags format', () => {
    const prompt = buildBatchPrompt(seqTask, 5)
    expect(prompt).toContain('tokens')
    expect(prompt).toContain('tags')
    expect(prompt).toContain('B-TYPE')
    expect(prompt).toContain('I-TYPE')
  })

  test('validateExample accepts valid sequence example', () => {
    expect(validateExample({
      tokens: ['John', 'went', 'to', 'Paris'],
      tags: ['B-PER', 'O', 'O', 'B-LOC']
    }, seqTask)).toBe(true)
  })

  test('validateExample rejects mismatched lengths', () => {
    expect(validateExample({
      tokens: ['John', 'went'],
      tags: ['B-PER']
    }, seqTask)).toBe(false)
  })

  test('validateExample rejects invalid tags', () => {
    expect(validateExample({
      tokens: ['John'],
      tags: ['B-INVALID']
    }, seqTask)).toBe(false)
  })

  test('validateExample rejects non-array tokens', () => {
    expect(validateExample({
      tokens: 'not an array',
      tags: ['O']
    }, seqTask)).toBe(false)
  })

  test('validateExample rejects empty tokens', () => {
    expect(validateExample({ tokens: [], tags: [] }, seqTask)).toBe(false)
  })
})

// ── NER Template ─────────────────────────────────────────

describe('NER template', () => {
  test('ner template exists', async () => {
    const templates = await listTemplates()
    const names = templates.map(t => t.name)
    expect(names).toContain('ner')
  })

  test('ner template is sequence-labeling type', async () => {
    const t = await loadTemplate('ner')
    expect(t.type).toBe('sequence-labeling')
    expect(t.labels).toContain('PER')
    expect(t.labels).toContain('ORG')
    expect(t.labels).toContain('LOC')
  })
})

// ── Exports ──────────────────────────────────────────────

describe('crf.js exports', () => {
  test('all expected functions are exported', () => {
    expect(typeof extractFeatures).toBe('function')
    expect(typeof featureHash).toBe('function')
    expect(typeof fnv1a).toBe('function')
    expect(typeof wordShape).toBe('function')
    expect(typeof viterbi).toBe('function')
    expect(typeof score).toBe('function')
    expect(typeof trainCRF).toBe('function')
    expect(typeof predictSequence).toBe('function')
    expect(typeof predictBatch).toBe('function')
    expect(typeof extractEntities).toBe('function')
    expect(typeof evaluateEntities).toBe('function')
    expect(typeof saveModel).toBe('function')
    expect(typeof loadModel).toBe('function')
    expect(typeof hasCRFModel).toBe('function')
    expect(typeof labelsToBIO).toBe('function')
    expect(typeof validateBIO).toBe('function')
  })
})
