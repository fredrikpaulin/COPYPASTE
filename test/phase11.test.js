import { test, expect, describe, beforeAll, afterAll } from 'bun:test'
import { join } from 'node:path'
import { mkdirSync, rmSync } from 'node:fs'

const TMP = join(import.meta.dir, 'test-tmp-phase11')

beforeAll(() => {
  mkdirSync(TMP, { recursive: true })
})

afterAll(() => {
  rmSync(TMP, { recursive: true, force: true })
})

// ── lib/ensemble.js — exports ────────────────────────────────

describe('ensemble — exports', () => {
  test('ALGORITHMS has 3 entries', async () => {
    const { ALGORITHMS } = await import('../lib/ensemble.js')
    expect(ALGORITHMS).toEqual(['logistic_regression', 'svm', 'random_forest'])
  })

  test('listEnsembleModels returns empty for nonexistent task', async () => {
    const { listEnsembleModels } = await import('../lib/ensemble.js')
    const models = await listEnsembleModels('nonexistent_task_xyz_' + Date.now())
    expect(models).toEqual([])
  })

  test('ensemblePredict throws when no ensemble models', async () => {
    const { ensemblePredict } = await import('../lib/ensemble.js')
    try {
      await ensemblePredict('nonexistent_task_xyz_' + Date.now(), ['test'])
      expect(true).toBe(false)
    } catch (e) {
      expect(e.message).toContain('No ensemble models found')
    }
  })

  test('predictWithThreshold exists', async () => {
    const { predictWithThreshold } = await import('../lib/ensemble.js')
    expect(typeof predictWithThreshold).toBe('function')
  })

  test('trainEnsembleModels exists', async () => {
    const { trainEnsembleModels } = await import('../lib/ensemble.js')
    expect(typeof trainEnsembleModels).toBe('function')
  })
})

// ── lib/experiment.js — hashDataset ──────────────────────────

describe('experiment — hashDataset', () => {
  test('returns 8-char hex string', async () => {
    const { hashDataset } = await import('../lib/experiment.js')
    const hash = hashDataset([{ text: 'hello', label: 'pos' }])
    expect(hash.length).toBe(8)
    expect(/^[0-9a-f]+$/.test(hash)).toBe(true)
  })

  test('same data produces same hash', async () => {
    const { hashDataset } = await import('../lib/experiment.js')
    const data = [{ text: 'hello', label: 'pos' }, { text: 'world', label: 'neg' }]
    expect(hashDataset(data)).toBe(hashDataset(data))
  })

  test('different data produces different hash', async () => {
    const { hashDataset } = await import('../lib/experiment.js')
    const a = hashDataset([{ text: 'hello', label: 'pos' }])
    const b = hashDataset([{ text: 'goodbye', label: 'neg' }])
    expect(a).not.toBe(b)
  })

  test('handles empty rows', async () => {
    const { hashDataset } = await import('../lib/experiment.js')
    const hash = hashDataset([{}])
    expect(hash.length).toBe(8)
  })

  test('order matters', async () => {
    const { hashDataset } = await import('../lib/experiment.js')
    const a = hashDataset([{ text: 'a', label: 'x' }, { text: 'b', label: 'y' }])
    const b = hashDataset([{ text: 'b', label: 'y' }, { text: 'a', label: 'x' }])
    expect(a).not.toBe(b)
  })
})

// ── lib/experiment.js — CRUD ─────────────────────────────────

describe('experiment — CRUD', () => {
  // Use a unique task name to avoid collisions
  const taskName = 'test_experiment_' + Date.now()

  test('recordExperiment returns an ID', async () => {
    const { recordExperiment } = await import('../lib/experiment.js')
    const id = recordExperiment({
      task: taskName,
      algorithm: 'logistic_regression',
      accuracy: 0.85,
      trainSize: 100,
      valSize: 25,
      dataHash: 'aabbccdd',
      featureMode: 'tfidf',
      labels: ['pos', 'neg'],
      durationMs: 1500
    })
    expect(typeof id).toBe('number')
    expect(id).toBeGreaterThan(0)
  })

  test('listExperiments returns recorded entries', async () => {
    const { listExperiments } = await import('../lib/experiment.js')
    const experiments = listExperiments(taskName)
    expect(experiments.length).toBeGreaterThanOrEqual(1)
    expect(experiments[0].task).toBe(taskName)
    expect(experiments[0].algorithm).toBe('logistic_regression')
    expect(experiments[0].accuracy).toBe(0.85)
  })

  test('getExperiment returns by ID', async () => {
    const { listExperiments, getExperiment } = await import('../lib/experiment.js')
    const experiments = listExperiments(taskName)
    const exp = getExperiment(experiments[0].id)
    expect(exp.task).toBe(taskName)
    expect(exp.labels).toEqual(['pos', 'neg'])
  })

  test('getExperiment returns null for missing ID', async () => {
    const { getExperiment } = await import('../lib/experiment.js')
    expect(getExperiment(999999)).toBeNull()
  })

  test('experimentStats returns aggregate info', async () => {
    const { experimentStats } = await import('../lib/experiment.js')
    const stats = experimentStats(taskName)
    expect(stats.total).toBeGreaterThanOrEqual(1)
    expect(stats.best_accuracy).toBe(0.85)
  })

  test('bestExperiment returns highest accuracy', async () => {
    const { recordExperiment, bestExperiment } = await import('../lib/experiment.js')
    // Add a better experiment
    recordExperiment({ task: taskName, algorithm: 'svm', accuracy: 0.92 })
    const best = bestExperiment(taskName)
    expect(best.accuracy).toBe(0.92)
    expect(best.algorithm).toBe('svm')
  })

  test('compareExperiments shows diff', async () => {
    const { listExperiments, compareExperiments } = await import('../lib/experiment.js')
    const experiments = listExperiments(taskName)
    expect(experiments.length).toBeGreaterThanOrEqual(2)

    const cmp = compareExperiments(experiments[1].id, experiments[0].id)
    expect(cmp.a).toBeTruthy()
    expect(cmp.b).toBeTruthy()
    expect(typeof cmp.diff.accuracyDelta).toBe('number')
    expect(cmp.diff.changes.length).toBeGreaterThan(0) // algorithm changed
  })

  test('compareExperiments throws for missing IDs', async () => {
    const { compareExperiments } = await import('../lib/experiment.js')
    try {
      compareExperiments(999998, 999999)
      expect(true).toBe(false)
    } catch (e) {
      expect(e.message).toContain('not found')
    }
  })

  test('clearExperiments removes entries', async () => {
    const { clearExperiments, listExperiments } = await import('../lib/experiment.js')
    const removed = clearExperiments(taskName)
    expect(removed).toBeGreaterThanOrEqual(2)
    expect(listExperiments(taskName).length).toBe(0)
  })

  test('listExperiments respects limit', async () => {
    const { recordExperiment, listExperiments } = await import('../lib/experiment.js')
    const t = taskName + '_limit'
    for (let i = 0; i < 5; i++) recordExperiment({ task: t, accuracy: i * 0.1 })
    const limited = listExperiments(t, { limit: 3 })
    expect(limited.length).toBe(3)
  })
})

// ── lib/experiment.js — hyperparams serialization ────────────

describe('experiment — hyperparams', () => {
  test('stores and retrieves JSON hyperparams', async () => {
    const { recordExperiment, getExperiment } = await import('../lib/experiment.js')
    const id = recordExperiment({
      task: 'hp_test_' + Date.now(),
      hyperparams: { C: 1.0, penalty: 'l2', max_iter: 500 }
    })
    const exp = getExperiment(id)
    expect(exp.hyperparams).toEqual({ C: 1.0, penalty: 'l2', max_iter: 500 })
  })

  test('null hyperparams stay null', async () => {
    const { recordExperiment, getExperiment } = await import('../lib/experiment.js')
    const id = recordExperiment({ task: 'hp_null_' + Date.now() })
    const exp = getExperiment(id)
    expect(exp.hyperparams).toBeNull()
  })
})
