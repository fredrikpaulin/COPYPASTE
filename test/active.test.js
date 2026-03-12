import { test, expect, describe, afterAll } from 'bun:test'
import { getUncertainExamples, loadHistory, saveIteration } from '../lib/active.js'
import { runTraining } from '../lib/train.js'
import { writeJsonl } from '../lib/data.js'
import { join } from 'node:path'
import { mkdir, rm } from 'node:fs/promises'

const ROOT = join(import.meta.dir, '..')
const TMP = join(ROOT, 'test-tmp-active')
const MODELS = join(ROOT, 'models')
const TASK_NAME = 'p5-active-test'

const trainData = [
  { text: 'I love this product so much', label: 'pos' },
  { text: 'Amazing quality and great', label: 'pos' },
  { text: 'Best purchase I ever made', label: 'pos' },
  { text: 'Highly recommend to everyone', label: 'pos' },
  { text: 'Great value for money here', label: 'pos' },
  { text: 'Wonderful and fantastic work', label: 'pos' },
  { text: 'Terrible product do not buy', label: 'neg' },
  { text: 'Awful quality really bad', label: 'neg' },
  { text: 'Waste of money honestly', label: 'neg' },
  { text: 'Horrible experience all around', label: 'neg' },
  { text: 'Very disappointing purchase', label: 'neg' },
  { text: 'Total garbage would return', label: 'neg' },
]

const valData = [
  { text: 'Love it works perfectly fine', label: 'pos' },
  { text: 'Works great for everything', label: 'pos' },
  { text: 'Completely broken and useless', label: 'neg' },
  { text: 'Very bad quality overall', label: 'neg' },
]

async function setup() {
  await mkdir(TMP, { recursive: true })
  await writeJsonl(join(TMP, 'train.jsonl'), trainData)
  await writeJsonl(join(TMP, 'val.jsonl'), valData)

  // Train a model to test against
  const task = { name: TASK_NAME, type: 'classification', labels: ['pos', 'neg'] }
  await runTraining(task, join(TMP, 'train.jsonl'), join(TMP, 'val.jsonl'), {
    onStdout: () => {}, onStderr: () => {}
  })
}

afterAll(async () => {
  await rm(TMP, { recursive: true, force: true }).catch(() => {})
  await rm(join(MODELS, TASK_NAME), { recursive: true, force: true }).catch(() => {})
})

// ── Uncertainty sampling ─────────────────────────────────

describe('getUncertainExamples', () => {
  test('returns predictions sorted by confidence ascending', async () => {
    await setup()
    const texts = [
      'This is amazing and wonderful',
      'It is okay I guess maybe',
      'Terrible awful horrible bad',
      'Pretty decent not great though',
    ]
    const results = await getUncertainExamples(TASK_NAME, texts, { topK: 4 })

    expect(results).toBeArray()
    expect(results.length).toBeGreaterThan(0)
    expect(results.length).toBeLessThanOrEqual(4)

    // Should be sorted by confidence ascending (most uncertain first)
    for (let i = 1; i < results.length; i++) {
      expect(results[i].confidence).toBeGreaterThanOrEqual(results[i - 1].confidence)
    }

    // Each result should have text, label, confidence
    for (const r of results) {
      expect(r.text).toBeDefined()
      expect(r.label).toBeDefined()
      expect(typeof r.confidence).toBe('number')
    }
  }, 15000)

  test('respects topK limit', async () => {
    const texts = ['good', 'bad', 'okay', 'meh', 'great']
    const results = await getUncertainExamples(TASK_NAME, texts, { topK: 2 })
    expect(results.length).toBeLessThanOrEqual(2)
  }, 15000)
})

// ── Iteration history ────────────────────────────────────

describe('iteration history', () => {
  test('starts with empty history', async () => {
    // Remove any existing history
    const histPath = join(MODELS, TASK_NAME, 'active_history.json')
    await rm(histPath, { force: true }).catch(() => {})

    const history = await loadHistory(TASK_NAME)
    expect(history.iterations).toEqual([])
  })

  test('saves and loads iterations', async () => {
    const histPath = join(MODELS, TASK_NAME, 'active_history.json')
    await rm(histPath, { force: true }).catch(() => {})

    await saveIteration(TASK_NAME, {
      examples_added: 10,
      accuracy_before: 0.85,
      method: 'llm_labeling'
    })

    await saveIteration(TASK_NAME, {
      examples_added: 5,
      accuracy_before: 0.88,
      method: 'manual'
    })

    const history = await loadHistory(TASK_NAME)
    expect(history.iterations.length).toBe(2)
    expect(history.iterations[0].examples_added).toBe(10)
    expect(history.iterations[0].timestamp).toBeDefined()
    expect(history.iterations[1].examples_added).toBe(5)
    expect(history.iterations[1].method).toBe('manual')
  })
})
