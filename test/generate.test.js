import { test, expect, describe, afterAll } from 'bun:test'
import {
  generate, preview,
  buildSystemPrompt, buildBatchPrompt, parseBatchResponse, validateExample,
  backoffMs
} from '../lib/generate.js'
import { join } from 'node:path'
import { rm } from 'node:fs/promises'
import { readJsonl } from '../lib/data.js'

const DATA_DIR = join(import.meta.dir, '..', 'data')

// ── buildSystemPrompt ─────────────────────────────────────

describe('buildSystemPrompt', () => {
  test('includes task description and type', () => {
    const task = { description: 'Classify things', type: 'classification', labels: ['a', 'b'] }
    const prompt = buildSystemPrompt(task)
    expect(prompt).toContain('Classify things')
    expect(prompt).toContain('classification')
  })

  test('includes labels for classification', () => {
    const task = { description: 'test', type: 'classification', labels: ['pos', 'neg', 'neutral'] }
    const prompt = buildSystemPrompt(task)
    expect(prompt).toContain('Labels: pos, neg, neutral')
  })

  test('includes fields for extraction', () => {
    const task = { description: 'test', type: 'extraction', fields: ['name', 'email'] }
    const prompt = buildSystemPrompt(task)
    expect(prompt).toContain('Fields to extract: name, email')
  })

  test('has no labels/fields line for regression', () => {
    const task = { description: 'test', type: 'regression' }
    const prompt = buildSystemPrompt(task)
    expect(prompt).not.toContain('Labels:')
    expect(prompt).not.toContain('Fields to extract:')
  })

  test('always starts with the data generator role', () => {
    const task = { description: 'test', type: 'regression' }
    const prompt = buildSystemPrompt(task)
    expect(prompt).toStartWith('You are a training data generator')
  })
})

// ── buildBatchPrompt ──────────────────────────────────────

describe('buildBatchPrompt', () => {
  test('requests the correct batch size', () => {
    const task = { type: 'classification', labels: ['a', 'b'], synthetic: { prompt: 'Generate example with {label}' } }
    const prompt = buildBatchPrompt(task, 15)
    expect(prompt).toContain('Generate exactly 15 training examples')
  })

  test('uses classification format', () => {
    const task = { type: 'classification', labels: ['a', 'b'], synthetic: { prompt: 'test' } }
    const prompt = buildBatchPrompt(task, 5)
    expect(prompt).toContain('"text"')
    expect(prompt).toContain('"label"')
  })

  test('uses extraction format', () => {
    const task = { type: 'extraction', fields: ['name'], synthetic: { prompt: 'test {field}' } }
    const prompt = buildBatchPrompt(task, 5)
    expect(prompt).toContain('"fields"')
  })

  test('uses regression format', () => {
    const task = { type: 'regression', synthetic: { prompt: 'test' } }
    const prompt = buildBatchPrompt(task, 5)
    expect(prompt).toContain('"value"')
  })

  test('includes label distribution instruction for classification', () => {
    const task = { type: 'classification', labels: ['pos', 'neg'], synthetic: { prompt: 'test' } }
    const prompt = buildBatchPrompt(task, 5)
    expect(prompt).toContain('Distribute labels roughly evenly')
  })

  test('replaces {label} placeholder in prompt', () => {
    const task = { type: 'classification', labels: ['happy', 'sad'], synthetic: { prompt: 'Write a {label} review' } }
    const prompt = buildBatchPrompt(task, 5)
    expect(prompt).toContain('happy or sad')
  })

  test('replaces {field} placeholder in prompt', () => {
    const task = { type: 'extraction', fields: ['name', 'phone'], synthetic: { prompt: 'Extract {field} from text' } }
    const prompt = buildBatchPrompt(task, 5)
    expect(prompt).toContain('name, phone')
  })

  test('ends with JSON-only instruction', () => {
    const task = { type: 'regression', synthetic: { prompt: 'test' } }
    const prompt = buildBatchPrompt(task, 5)
    expect(prompt).toContain('Respond with ONLY a valid JSON array')
  })
})

// ── parseBatchResponse ────────────────────────────────────

describe('parseBatchResponse', () => {
  test('parses clean JSON array', () => {
    const result = parseBatchResponse('[{"text": "hello", "label": "a"}, {"text": "world", "label": "b"}]')
    expect(result).toHaveLength(2)
    expect(result[0].text).toBe('hello')
  })

  test('extracts JSON from markdown code fences', () => {
    const result = parseBatchResponse('```json\n[{"text": "test", "label": "a"}]\n```')
    expect(result).toHaveLength(1)
  })

  test('extracts JSON with surrounding text', () => {
    const result = parseBatchResponse('Here are the examples:\n[{"text": "x", "label": "y"}]\nDone!')
    expect(result).toHaveLength(1)
  })

  test('handles multiline JSON', () => {
    const result = parseBatchResponse('[\n  {"text": "a", "label": "b"},\n  {"text": "c", "label": "d"}\n]')
    expect(result).toHaveLength(2)
  })

  test('throws on missing array', () => {
    expect(() => parseBatchResponse('no json here')).toThrow('No JSON array found')
  })

  test('throws on invalid JSON', () => {
    expect(() => parseBatchResponse('[{invalid json}]')).toThrow()
  })
})

// ── validateExample ───────────────────────────────────────

describe('validateExample', () => {
  test('accepts valid classification example', () => {
    const task = { type: 'classification', labels: ['pos', 'neg'] }
    expect(validateExample({ text: 'great', label: 'pos' }, task)).toBe(true)
  })

  test('rejects classification with wrong label', () => {
    const task = { type: 'classification', labels: ['pos', 'neg'] }
    expect(validateExample({ text: 'great', label: 'unknown' }, task)).toBe(false)
  })

  test('rejects missing text', () => {
    const task = { type: 'classification', labels: ['a'] }
    expect(validateExample({ label: 'a' }, task)).toBe(false)
  })

  test('rejects empty text', () => {
    const task = { type: 'classification', labels: ['a'] }
    expect(validateExample({ text: '  ', label: 'a' }, task)).toBe(false)
  })

  test('rejects non-object', () => {
    expect(validateExample(null, { type: 'classification' })).toBe(false)
    expect(validateExample('string', { type: 'classification' })).toBe(false)
  })

  test('accepts valid extraction example', () => {
    const task = { type: 'extraction', fields: ['name'] }
    expect(validateExample({ text: 'hello', fields: { name: 'John' } }, task)).toBe(true)
  })

  test('rejects extraction without fields object', () => {
    const task = { type: 'extraction', fields: ['name'] }
    expect(validateExample({ text: 'hello' }, task)).toBe(false)
  })

  test('accepts valid regression example', () => {
    const task = { type: 'regression' }
    expect(validateExample({ text: 'hello', value: 3.5 }, task)).toBe(true)
  })

  test('rejects regression with string value', () => {
    const task = { type: 'regression' }
    expect(validateExample({ text: 'hello', value: 'three' }, task)).toBe(false)
  })
})

// ── backoffMs ─────────────────────────────────────────────

describe('backoffMs', () => {
  test('increases with attempt number', () => {
    const samples = Array.from({ length: 100 }, () => backoffMs(0, 1000, 60000))
    const avg0 = samples.reduce((a, b) => a + b) / samples.length

    const samples2 = Array.from({ length: 100 }, () => backoffMs(2, 1000, 60000))
    const avg2 = samples2.reduce((a, b) => a + b) / samples2.length

    expect(avg2).toBeGreaterThan(avg0)
  })

  test('respects max ceiling', () => {
    for (let i = 0; i < 50; i++) {
      expect(backoffMs(10, 1000, 5000)).toBeLessThanOrEqual(5000)
    }
  })

  test('always returns positive value', () => {
    for (let i = 0; i < 50; i++) {
      expect(backoffMs(i, 1000, 60000)).toBeGreaterThan(0)
    }
  })
})

// ── generate (integration with mock server) ───────────────

describe('generate (mock API)', () => {
  let server
  let port

  const startMockServer = (responses) => {
    let callIdx = 0
    server = Bun.serve({
      port: 0,
      fetch(req) {
        const batch = responses[Math.min(callIdx, responses.length - 1)]
        callIdx++
        return new Response(JSON.stringify({
          content: [{ text: JSON.stringify(batch) }]
        }), { headers: { 'Content-Type': 'application/json' } })
      }
    })
    port = server.port
  }

  afterAll(() => {
    server?.stop()
    rm(join(DATA_DIR, 'mock-gen_synthetic.jsonl'), { force: true }).catch(() => {})
  })

  test('generates data via mock API and writes JSONL', async () => {
    startMockServer([
      [{ text: 'great', label: 'pos' }, { text: 'bad', label: 'neg' }],
      [{ text: 'ok', label: 'neutral' }, { text: 'fine', label: 'pos' }]
    ])

    const origFetch = globalThis.fetch
    globalThis.fetch = (url, opts) => origFetch(`http://localhost:${port}/v1/messages`, opts)

    const task = {
      name: 'mock-gen', type: 'classification', labels: ['pos', 'neg', 'neutral'],
      description: 'test', synthetic: { count: 4, prompt: 'test', batchSize: 2, model: 'test-model' }
    }

    const progressCalls = []
    const result = await generate(task, {
      apiKey: 'test-key',
      onProgress: (cur, total) => progressCalls.push({ cur, total })
    })

    globalThis.fetch = origFetch

    expect(result.count).toBe(4)
    expect(result.dropped).toBe(0)
    expect(progressCalls.length).toBe(2)
  })

  test('drops invalid examples and reports count', async () => {
    server?.stop()
    // One valid, one with wrong label, one missing text
    const batch = [
      { text: 'good', label: 'pos' },
      { text: 'bad', label: 'INVALID_LABEL' },
      { label: 'pos' }
    ]
    startMockServer([batch])

    const origFetch = globalThis.fetch
    globalThis.fetch = (url, opts) => origFetch(`http://localhost:${port}/v1/messages`, opts)

    const task = {
      name: 'mock-gen', type: 'classification', labels: ['pos', 'neg'],
      description: 'test', synthetic: { count: 3, prompt: 'test', batchSize: 3 }
    }

    let droppedReport = 0
    const result = await generate(task, {
      apiKey: 'key',
      onProgress: () => {},
      onDropped: n => { droppedReport = n }
    })

    globalThis.fetch = origFetch

    expect(result.count).toBe(1)
    expect(result.dropped).toBe(2)
    expect(droppedReport).toBe(2)
  })

  test('retries on 429 and reports via callback', async () => {
    server?.stop()
    let callCount = 0
    server = Bun.serve({
      port: 0,
      fetch() {
        callCount++
        if (callCount <= 1) {
          return new Response('rate limited', { status: 429 })
        }
        return new Response(JSON.stringify({
          content: [{ text: '[{"text":"ok","label":"a"}]' }]
        }))
      }
    })
    port = server.port

    const origFetch = globalThis.fetch
    globalThis.fetch = (url, opts) => origFetch(`http://localhost:${port}/v1/messages`, opts)

    const task = {
      name: 'mock-gen', type: 'classification', labels: ['a', 'b'],
      description: 'test', synthetic: { count: 1, prompt: 'test', batchSize: 1 }
    }

    const retries = []
    const result = await generate(task, {
      apiKey: 'key',
      onProgress: () => {},
      onRetry: info => retries.push(info)
    })

    globalThis.fetch = origFetch

    expect(result.count).toBe(1)
    expect(retries.length).toBe(1)
    expect(retries[0].status).toBe(429)
  })

  test('fails after max retries', async () => {
    server?.stop()
    server = Bun.serve({
      port: 0,
      fetch() { return new Response('overloaded', { status: 529, headers: { 'retry-after': '0' } }) }
    })
    port = server.port

    const origFetch = globalThis.fetch
    globalThis.fetch = (url, opts) => origFetch(`http://localhost:${port}/v1/messages`, opts)

    const task = {
      name: 'mock-err', type: 'classification', labels: ['a', 'b'],
      description: 'test', synthetic: { count: 1, prompt: 'test', batchSize: 1 }
    }

    try {
      await generate(task, { apiKey: 'key', onProgress: () => {} })
      expect(true).toBe(false)
    } catch (e) {
      expect(e.message).toContain('529')
      expect(e.message).toContain('attempts')
    }

    globalThis.fetch = origFetch
  })
})
