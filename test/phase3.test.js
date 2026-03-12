import { test, expect, describe, afterAll } from 'bun:test'
import { runTraining } from '../lib/train.js'
import { writeJsonl } from '../lib/data.js'
import { join } from 'node:path'
import { mkdir, rm } from 'node:fs/promises'

const ROOT = join(import.meta.dir, '..')
const TMP = join(ROOT, 'test-tmp-phase3')
const MODELS = join(ROOT, 'models')

const classificationData = [
  { text: 'I love this product', label: 'pos' },
  { text: 'Amazing quality', label: 'pos' },
  { text: 'Best purchase ever', label: 'pos' },
  { text: 'Highly recommend', label: 'pos' },
  { text: 'Great value', label: 'pos' },
  { text: 'Wonderful experience', label: 'pos' },
  { text: 'Fantastic service', label: 'pos' },
  { text: 'Really good stuff', label: 'pos' },
  { text: 'Terrible product', label: 'neg' },
  { text: 'Awful quality', label: 'neg' },
  { text: 'Waste of money', label: 'neg' },
  { text: 'Horrible experience', label: 'neg' },
  { text: 'Very disappointing', label: 'neg' },
  { text: 'Total garbage', label: 'neg' },
  { text: 'Would not recommend', label: 'neg' },
  { text: 'Broken on arrival', label: 'neg' },
]

const valData = [
  { text: 'Love it so much', label: 'pos' },
  { text: 'Works perfectly', label: 'pos' },
  { text: 'Completely broken', label: 'neg' },
  { text: 'Very bad quality', label: 'neg' },
]

// Extraction data: each field needs both present (truthy) and absent (falsy) examples
const extractionTrain = [
  { text: 'Call John at 555-0123', fields: { name: 'John', phone: '555-0123' } },
  { text: 'The office number is 555-4444', fields: { name: null, phone: '555-4444' } },
  { text: 'Reach Bob at 555-9999', fields: { name: 'Bob', phone: '555-9999' } },
  { text: 'Please check the schedule for today', fields: { name: null, phone: null } },
  { text: 'Message Sam at 555-1111 urgently', fields: { name: 'Sam', phone: '555-1111' } },
  { text: 'The weather looks nice outside', fields: { name: null, phone: null } },
  { text: 'Find Dave at 555-2222', fields: { name: 'Dave', phone: '555-2222' } },
  { text: 'Call the front desk at 555-7777', fields: { name: null, phone: '555-7777' } },
]

const extractionVal = [
  { text: 'Ring Mike at 555-3333', fields: { name: 'Mike', phone: '555-3333' } },
  { text: 'Remember to bring the files tomorrow', fields: { name: null, phone: null } },
]

// Task names used by tests — cleaned up after
const TEST_TASKS = ['p3-algo-lr', 'p3-algo-svm', 'p3-algo-rf', 'p3-compare', 'p3-search', 'p3-extract']

async function setup() {
  await mkdir(TMP, { recursive: true })
  await writeJsonl(join(TMP, 'train.jsonl'), classificationData)
  await writeJsonl(join(TMP, 'val.jsonl'), valData)
  await writeJsonl(join(TMP, 'ext_train.jsonl'), extractionTrain)
  await writeJsonl(join(TMP, 'ext_val.jsonl'), extractionVal)
}

afterAll(async () => {
  await rm(TMP, { recursive: true, force: true }).catch(() => {})
  for (const t of TEST_TASKS) {
    await rm(join(MODELS, t), { recursive: true, force: true }).catch(() => {})
  }
})

// ── Algorithm selection ──────────────────────────────────

describe('algorithm selection', () => {
  test('trains with logistic_regression', async () => {
    await setup()
    const task = { name: 'p3-algo-lr', type: 'classification', labels: ['pos', 'neg'] }
    const result = await runTraining(task, join(TMP, 'train.jsonl'), join(TMP, 'val.jsonl'), {
      algorithm: 'logistic_regression',
      onStdout: () => {}, onStderr: () => {}
    })
    expect(result.stdout).toContain('Algorithm: logistic_regression')
    expect(result.stdout).toContain('Validation Accuracy')
  }, 30000)

  test('trains with svm', async () => {
    await setup()
    const task = { name: 'p3-algo-svm', type: 'classification', labels: ['pos', 'neg'] }
    const result = await runTraining(task, join(TMP, 'train.jsonl'), join(TMP, 'val.jsonl'), {
      algorithm: 'svm',
      onStdout: () => {}, onStderr: () => {}
    })
    expect(result.stdout).toContain('Algorithm: svm')
    expect(result.stdout).toContain('Validation Accuracy')
  }, 30000)

  test('trains with random_forest', async () => {
    await setup()
    const task = { name: 'p3-algo-rf', type: 'classification', labels: ['pos', 'neg'] }
    const result = await runTraining(task, join(TMP, 'train.jsonl'), join(TMP, 'val.jsonl'), {
      algorithm: 'random_forest',
      onStdout: () => {}, onStderr: () => {}
    })
    expect(result.stdout).toContain('Algorithm: random_forest')
    expect(result.stdout).toContain('Validation Accuracy')
  }, 30000)
})

// ── Model comparison ─────────────────────────────────────

describe('model comparison', () => {
  test('compares multiple algorithms and saves best', async () => {
    await setup()
    const task = { name: 'p3-compare', type: 'classification', labels: ['pos', 'neg'] }
    const result = await runTraining(task, join(TMP, 'train.jsonl'), join(TMP, 'val.jsonl'), {
      compare: true,
      onStdout: () => {}, onStderr: () => {}
    })
    expect(result.stdout).toContain('Comparing 3 algorithms')
    expect(result.stdout).toContain('COMPARISON RESULTS')
    expect(result.stdout).toContain('__COMPARE_RESULTS__:')

    const meta = await Bun.file(join(result.modelDir, 'meta.json')).json()
    expect(meta.comparison).toBeArray()
    expect(meta.comparison.length).toBe(3)
    expect(meta.algorithm).toBeDefined()
  }, 60000)
})

// ── Hyperparameter search ────────────────────────────────

describe('hyperparameter search', () => {
  test('runs grid search and saves best model', async () => {
    await setup()
    const task = { name: 'p3-search', type: 'classification', labels: ['pos', 'neg'] }
    const grid = { C: [0.1, 1.0], max_iter: [500] }
    const result = await runTraining(task, join(TMP, 'train.jsonl'), join(TMP, 'val.jsonl'), {
      algorithm: 'logistic_regression',
      search: true,
      grid,
      onStdout: () => {}, onStderr: () => {}
    })
    expect(result.stdout).toContain('Hyperparameter search')
    expect(result.stdout).toContain('2 combinations')
    expect(result.stdout).toContain('SEARCH RESULTS')
    expect(result.stdout).toContain('__SEARCH_RESULTS__:')

    const meta = await Bun.file(join(result.modelDir, 'meta.json')).json()
    expect(meta.search_results).toBeArray()
    expect(meta.params).toBeDefined()
  }, 60000)
})

// ── Extraction training ──────────────────────────────────

describe('extraction training', () => {
  test('trains per-field extractors', async () => {
    await setup()
    const task = { name: 'p3-extract', type: 'extraction', fields: ['name', 'phone'] }
    const result = await runTraining(task, join(TMP, 'ext_train.jsonl'), join(TMP, 'ext_val.jsonl'), {
      onStdout: () => {}, onStderr: () => {}
    })
    expect(result.stdout).toContain('Training extractor for field: name')
    expect(result.stdout).toContain('Training extractor for field: phone')
    expect(result.stdout).toContain('detection accuracy')

    const meta = await Bun.file(join(result.modelDir, 'meta.json')).json()
    expect(meta.task_type).toBe('extraction')
    expect(meta.fields).toEqual(['name', 'phone'])
    expect(meta.field_accuracies).toBeDefined()
  }, 30000)
})
