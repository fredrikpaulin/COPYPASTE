import { test, expect, describe, beforeEach, afterEach } from 'bun:test'
import { loadTask, listTasks, saveTask, validate, loadSchema } from '../lib/task.js'
import { join } from 'node:path'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'

let schema

beforeEach(async () => {
  schema = await loadSchema()
})

// ── validate ──────────────────────────────────────────────

describe('validate', () => {
  test('accepts a valid classification task', () => {
    const task = {
      name: 'sentiment',
      type: 'classification',
      description: 'Classify sentiment',
      labels: ['positive', 'negative']
    }
    expect(validate(task, schema)).toEqual([])
  })

  test('accepts a valid extraction task', () => {
    const task = {
      name: 'ner-extract',
      type: 'extraction',
      description: 'Extract entities',
      fields: ['name', 'email']
    }
    expect(validate(task, schema)).toEqual([])
  })

  test('accepts a valid regression task', () => {
    const task = {
      name: 'score-predict',
      type: 'regression',
      description: 'Predict review score'
    }
    expect(validate(task, schema)).toEqual([])
  })

  test('rejects missing required fields', () => {
    const errors = validate({}, schema)
    const paths = errors.map(e => e.path)
    expect(paths).toContain('name')
    expect(paths).toContain('type')
    expect(paths).toContain('description')
  })

  test('rejects invalid name pattern', () => {
    const task = { name: 'Bad Name!', type: 'classification', description: 'x', labels: ['a', 'b'] }
    const errors = validate(task, schema)
    expect(errors.some(e => e.path === 'name')).toBe(true)
  })

  test('accepts valid name patterns', () => {
    for (const name of ['my-task', 'task_1', 'abc123', 'a-b_c']) {
      const task = { name, type: 'regression', description: 'x' }
      expect(validate(task, schema)).toEqual([])
    }
  })

  test('rejects invalid type enum', () => {
    const task = { name: 'test', type: 'invalid', description: 'x' }
    const errors = validate(task, schema)
    expect(errors.some(e => e.path === 'type')).toBe(true)
  })

  test('requires labels for classification', () => {
    const task = { name: 'test', type: 'classification', description: 'x' }
    const errors = validate(task, schema)
    expect(errors.some(e => e.path === 'labels')).toBe(true)
  })

  test('requires labels for sequence-labeling', () => {
    const task = { name: 'test', type: 'sequence-labeling', description: 'x' }
    const errors = validate(task, schema)
    expect(errors.some(e => e.path === 'labels')).toBe(true)
  })

  test('requires fields for extraction', () => {
    const task = { name: 'test', type: 'extraction', description: 'x' }
    const errors = validate(task, schema)
    expect(errors.some(e => e.path === 'fields')).toBe(true)
  })

  test('does not require labels for regression', () => {
    const task = { name: 'test', type: 'regression', description: 'x' }
    expect(validate(task, schema)).toEqual([])
  })

  test('rejects labels with fewer than 2 items', () => {
    const task = { name: 'test', type: 'classification', description: 'x', labels: ['one'] }
    const errors = validate(task, schema)
    expect(errors.some(e => e.path === 'labels')).toBe(true)
  })

  test('rejects non-object input', () => {
    expect(validate('string', schema).length).toBeGreaterThan(0)
    expect(validate(null, schema).length).toBeGreaterThan(0)
    expect(validate([], schema).length).toBeGreaterThan(0)
  })
})

// ── loadTask ──────────────────────────────────────────────

describe('loadTask', () => {
  test('loads the example sentiment task', async () => {
    const task = await loadTask('sentiment')
    expect(task.name).toBe('sentiment')
    expect(task.type).toBe('classification')
    expect(task.labels).toEqual(['positive', 'negative', 'neutral'])
    expect(task.description).toBeTypeOf('string')
  })

  test('applies defaults to loaded task', async () => {
    const task = await loadTask('sentiment')
    // synthetic.model default
    expect(task.synthetic.model).toBe('claude-sonnet-4-20250514')
    // training.script default
    expect(task.training.script).toBe('scripts/train.py')
  })

  test('throws for nonexistent task', async () => {
    expect(loadTask('does-not-exist')).rejects.toThrow('not found')
  })
})

// ── listTasks ─────────────────────────────────────────────

describe('listTasks', () => {
  test('returns array including sentiment', async () => {
    const tasks = await listTasks()
    expect(Array.isArray(tasks)).toBe(true)
    expect(tasks).toContain('sentiment')
  })

  test('does not include file extensions', async () => {
    const tasks = await listTasks()
    for (const t of tasks) {
      expect(t).not.toContain('.json')
    }
  })
})

// ── saveTask ──────────────────────────────────────────────

describe('saveTask', () => {
  test('rejects invalid task on save', async () => {
    const bad = { name: 'Bad!', type: 'invalid', description: '' }
    expect(saveTask(bad)).rejects.toThrow()
  })

  test('saves a valid task to disk', async () => {
    const task = {
      name: 'test-save-task',
      type: 'classification',
      description: 'Test save',
      labels: ['a', 'b'],
      synthetic: { count: 10, prompt: 'test' }
    }
    const path = await saveTask(task)
    expect(path).toContain('test-save-task.json')

    // Verify it round-trips
    const loaded = await loadTask('test-save-task')
    expect(loaded.name).toBe('test-save-task')
    expect(loaded.labels).toEqual(['a', 'b'])

    // Clean up
    await rm(path)
  })
})

// ── loadSchema ────────────────────────────────────────────

describe('loadSchema', () => {
  test('returns a valid JSON Schema object', async () => {
    const s = await loadSchema()
    expect(s.title).toBe('DistillationTask')
    expect(s.type).toBe('object')
    expect(s.properties).toBeDefined()
    expect(s.required).toContain('name')
  })
})
