import { test, expect, describe, beforeAll, afterAll } from 'bun:test'
import { join } from 'node:path'
import { mkdirSync, rmSync, writeFileSync } from 'node:fs'

const TMP = join(import.meta.dir, 'test-tmp-phase12')

beforeAll(() => {
  mkdirSync(TMP, { recursive: true })
})

afterAll(() => {
  rmSync(TMP, { recursive: true, force: true })
})

// ── lib/transformer.js — exports ─────────────────────────────

describe('transformer — exports', () => {
  test('exports all expected functions', async () => {
    const mod = await import('../lib/transformer.js')
    expect(typeof mod.checkDeps).toBe('function')
    expect(typeof mod.detectDevice).toBe('function')
    expect(typeof mod.listModelPresets).toBe('function')
    expect(typeof mod.trainTransformer).toBe('function')
    expect(typeof mod.predictTransformer).toBe('function')
    expect(typeof mod.hasTransformerModel).toBe('function')
  })

  test('MODEL_PRESETS has expected keys', async () => {
    const { MODEL_PRESETS } = await import('../lib/transformer.js')
    expect(Object.keys(MODEL_PRESETS)).toContain('distilbert')
    expect(Object.keys(MODEL_PRESETS)).toContain('tinybert')
    expect(Object.keys(MODEL_PRESETS)).toContain('bert-base')
    expect(Object.keys(MODEL_PRESETS)).toContain('roberta')
    expect(Object.keys(MODEL_PRESETS)).toContain('minilm')
    expect(Object.keys(MODEL_PRESETS).length).toBe(5)
  })
})

// ── listModelPresets ─────────────────────────────────────────

describe('transformer — listModelPresets', () => {
  test('returns array of presets with required fields', async () => {
    const { listModelPresets } = await import('../lib/transformer.js')
    const presets = listModelPresets()
    expect(Array.isArray(presets)).toBe(true)
    expect(presets.length).toBe(5)
    for (const p of presets) {
      expect(typeof p.key).toBe('string')
      expect(typeof p.name).toBe('string')
      expect(typeof p.params).toBe('string')
      expect(typeof p.description).toBe('string')
    }
  })

  test('distilbert preset has correct details', async () => {
    const { listModelPresets } = await import('../lib/transformer.js')
    const presets = listModelPresets()
    const db = presets.find(p => p.key === 'distilbert')
    expect(db).toBeTruthy()
    expect(db.name).toBe('distilbert-base-uncased')
    expect(db.params).toBe('66M')
  })

  test('tinybert is the smallest model', async () => {
    const { listModelPresets } = await import('../lib/transformer.js')
    const presets = listModelPresets()
    const tb = presets.find(p => p.key === 'tinybert')
    expect(tb.params).toBe('14M')
  })
})

// ── checkDeps ────────────────────────────────────────────────

describe('transformer — checkDeps', () => {
  test('returns object with torch and transformers keys', async () => {
    const { checkDeps } = await import('../lib/transformer.js')
    const deps = await checkDeps()
    expect(typeof deps).toBe('object')
    expect('torch' in deps).toBe(true)
    expect('transformers' in deps).toBe(true)
    // Values should be booleans
    expect(typeof deps.torch).toBe('boolean')
    expect(typeof deps.transformers).toBe('boolean')
  })
})

// ── detectDevice ─────────────────────────────────────────────

describe('transformer — detectDevice', () => {
  test('returns object with device and info fields', async () => {
    const { detectDevice, checkDeps } = await import('../lib/transformer.js')
    const deps = await checkDeps()

    const result = await detectDevice()
    expect(typeof result).toBe('object')
    expect('device' in result).toBe(true)
    expect('info' in result).toBe(true)

    if (deps.torch) {
      // If torch is available, device should be cuda, mps, or cpu
      expect(['cuda', 'mps', 'cpu']).toContain(result.device)
    }
  })
})

// ── hasTransformerModel ──────────────────────────────────────

describe('transformer — hasTransformerModel', () => {
  test('returns false for nonexistent task', async () => {
    const { hasTransformerModel } = await import('../lib/transformer.js')
    const result = await hasTransformerModel('nonexistent_task_xyz_' + Date.now())
    expect(result).toBe(false)
  })
})

// ── scripts/train_transformer.py — utility modes ─────────────

describe('train_transformer.py — utility modes', () => {
  test('--check-deps returns valid JSON', async () => {
    const proc = Bun.spawn(['python3', 'scripts/train_transformer.py', '--check-deps'], {
      cwd: join(import.meta.dir, '..'),
      stdout: 'pipe',
      stderr: 'pipe'
    })
    const text = await new Response(proc.stdout).text()
    await proc.exited
    const result = JSON.parse(text.trim())
    expect(typeof result.torch).toBe('boolean')
    expect(typeof result.transformers).toBe('boolean')
  })

  test('--list-models returns presets with all expected keys', async () => {
    const proc = Bun.spawn(['python3', 'scripts/train_transformer.py', '--list-models'], {
      cwd: join(import.meta.dir, '..'),
      stdout: 'pipe',
      stderr: 'pipe'
    })
    const text = await new Response(proc.stdout).text()
    await proc.exited
    const result = JSON.parse(text.trim())
    expect(result.distilbert).toBeTruthy()
    expect(result.tinybert).toBeTruthy()
    expect(result['bert-base']).toBeTruthy()
    expect(result.roberta).toBeTruthy()
    expect(result.minilm).toBeTruthy()

    // Each preset should have name, max_length, default_lr, default_epochs, params
    for (const [key, preset] of Object.entries(result)) {
      expect(typeof preset.name).toBe('string')
      expect(typeof preset.max_length).toBe('number')
      expect(typeof preset.default_lr).toBe('number')
      expect(typeof preset.default_epochs).toBe('number')
      expect(typeof preset.params).toBe('string')
    }
  })

  test('--detect-device returns valid JSON', async () => {
    const proc = Bun.spawn(['python3', 'scripts/train_transformer.py', '--detect-device'], {
      cwd: join(import.meta.dir, '..'),
      stdout: 'pipe',
      stderr: 'pipe'
    })
    const text = await new Response(proc.stdout).text()
    await proc.exited
    const result = JSON.parse(text.trim())
    expect('device' in result).toBe(true)
    expect('info' in result).toBe(true)
  })

  test('missing --train flag in training mode shows error', async () => {
    const proc = Bun.spawn(['python3', 'scripts/train_transformer.py', '--val', 'x', '--output', 'y'], {
      cwd: join(import.meta.dir, '..'),
      stdout: 'pipe',
      stderr: 'pipe'
    })
    await proc.exited
    expect(proc.exitCode).not.toBe(0)
  })
})

// ── MODEL_PRESETS consistency between JS and Python ──────────

describe('transformer — JS/Python preset consistency', () => {
  test('JS and Python have same model keys', async () => {
    const { MODEL_PRESETS } = await import('../lib/transformer.js')
    const jsKeys = Object.keys(MODEL_PRESETS).sort()

    const proc = Bun.spawn(['python3', 'scripts/train_transformer.py', '--list-models'], {
      cwd: join(import.meta.dir, '..'),
      stdout: 'pipe',
      stderr: 'pipe'
    })
    const text = await new Response(proc.stdout).text()
    await proc.exited
    const pyPresets = JSON.parse(text.trim())
    const pyKeys = Object.keys(pyPresets).sort()

    expect(jsKeys).toEqual(pyKeys)
  })

  test('JS and Python agree on model names', async () => {
    const { MODEL_PRESETS } = await import('../lib/transformer.js')

    const proc = Bun.spawn(['python3', 'scripts/train_transformer.py', '--list-models'], {
      cwd: join(import.meta.dir, '..'),
      stdout: 'pipe',
      stderr: 'pipe'
    })
    const text = await new Response(proc.stdout).text()
    await proc.exited
    const pyPresets = JSON.parse(text.trim())

    for (const key of Object.keys(MODEL_PRESETS)) {
      expect(MODEL_PRESETS[key].name).toBe(pyPresets[key].name)
    }
  })
})

// ── trainTransformer error handling ──────────────────────────

describe('transformer — trainTransformer', () => {
  test('rejects with clear error when deps missing and train data invalid', async () => {
    const { trainTransformer } = await import('../lib/transformer.js')

    // Use nonexistent paths — should fail during Python execution
    const task = { name: 'test-nonexistent', type: 'classification', labels: ['a', 'b'] }
    try {
      await trainTransformer(task, '/nonexistent/train.jsonl', '/nonexistent/val.jsonl')
      // Should not reach here
      expect(true).toBe(false)
    } catch (e) {
      expect(e.message).toContain('exited with code')
    }
  })
})

// ── predictTransformer error handling ────────────────────────

describe('transformer — predictTransformer', () => {
  test('rejects when no model directory exists', async () => {
    const { predictTransformer } = await import('../lib/transformer.js')
    try {
      await predictTransformer('nonexistent_task_' + Date.now(), ['test'])
      expect(true).toBe(false)
    } catch (e) {
      expect(e).toBeTruthy()
    }
  })
})
