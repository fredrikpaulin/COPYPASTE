import { test, expect, describe, afterAll } from 'bun:test'
import { generate, buildSystemPrompt, buildBatchPrompt, parseBatchResponse } from '../lib/generate.js'
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
    const task = {
      type: 'classification',
      labels: ['a', 'b'],
      synthetic: { prompt: 'Generate example with {label}' }
    }
    const prompt = buildBatchPrompt(task, 15)
    expect(prompt).toContain('Generate exactly 15 training examples')
  })

  test('uses classification format', () => {
    const task = {
      type: 'classification',
      labels: ['a', 'b'],
      synthetic: { prompt: 'test' }
    }
    const prompt = buildBatchPrompt(task, 5)
    expect(prompt).toContain('"text"')
    expect(prompt).toContain('"label"')
  })

  test('uses extraction format', () => {
    const task = {
      type: 'extraction',
      fields: ['name'],
      synthetic: { prompt: 'test {field}' }
    }
    const prompt = buildBatchPrompt(task, 5)
    expect(prompt).toContain('"fields"')
  })

  test('uses regression format', () => {
    const task = {
      type: 'regression',
      synthetic: { prompt: 'test' }
    }
    const prompt = buildBatchPrompt(task, 5)
    expect(prompt).toContain('"value"')
  })

  test('includes label distribution instruction for classification', () => {
    const task = {
      type: 'classification',
      labels: ['pos', 'neg'],
      synthetic: { prompt: 'test' }
    }
    const prompt = buildBatchPrompt(task, 5)
    expect(prompt).toContain('Distribute labels roughly evenly')
    expect(prompt).toContain('pos, neg')
  })

  test('replaces {label} placeholder in prompt', () => {
    const task = {
      type: 'classification',
      labels: ['happy', 'sad'],
      synthetic: { prompt: 'Write a {label} review' }
    }
    const prompt = buildBatchPrompt(task, 5)
    expect(prompt).toContain('happy or sad')
  })

  test('replaces {field} placeholder in prompt', () => {
    const task = {
      type: 'extraction',
      fields: ['name', 'phone'],
      synthetic: { prompt: 'Extract {field} from text' }
    }
    const prompt = buildBatchPrompt(task, 5)
    expect(prompt).toContain('name, phone')
  })

  test('ends with JSON-only instruction', () => {
    const task = {
      type: 'regression',
      synthetic: { prompt: 'test' }
    }
    const prompt = buildBatchPrompt(task, 5)
    expect(prompt).toContain('Respond with ONLY a valid JSON array')
  })
})

// ── parseBatchResponse ────────────────────────────────────

describe('parseBatchResponse', () => {
  test('parses clean JSON array', () => {
    const text = '[{"text": "hello", "label": "a"}, {"text": "world", "label": "b"}]'
    const result = parseBatchResponse(text)
    expect(result).toHaveLength(2)
    expect(result[0].text).toBe('hello')
  })

  test('extracts JSON from markdown code fences', () => {
    const text = '```json\n[{"text": "test", "label": "a"}]\n```'
    const result = parseBatchResponse(text)
    expect(result).toHaveLength(1)
    expect(result[0].label).toBe('a')
  })

  test('extracts JSON with surrounding text', () => {
    const text = 'Here are the examples:\n[{"text": "x", "label": "y"}]\nDone!'
    const result = parseBatchResponse(text)
    expect(result).toHaveLength(1)
  })

  test('handles multiline JSON', () => {
    const text = `[
  {"text": "line 1", "label": "a"},
  {"text": "line 2", "label": "b"}
]`
    const result = parseBatchResponse(text)
    expect(result).toHaveLength(2)
  })

  test('throws on missing array', () => {
    expect(() => parseBatchResponse('no json here')).toThrow('No JSON array found')
  })

  test('throws on invalid JSON', () => {
    expect(() => parseBatchResponse('[{invalid json}]')).toThrow()
  })
})

// ── generate (integration with mock server) ───────────────

describe('generate (mock API)', () => {
  let server
  let port

  // Spin up a local server that mimics the Claude Messages API
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
    // Clean up any test data files
    rm(join(DATA_DIR, 'mock-gen_synthetic.jsonl'), { force: true }).catch(() => {})
  })

  test('generates data via mock API and writes JSONL', async () => {
    const batch1 = [
      { text: 'great', label: 'pos' },
      { text: 'bad', label: 'neg' }
    ]
    const batch2 = [
      { text: 'ok', label: 'neutral' },
      { text: 'fine', label: 'pos' }
    ]
    startMockServer([batch1, batch2])

    // Monkey-patch the API URL by overriding fetch
    const origFetch = globalThis.fetch
    globalThis.fetch = (url, opts) => {
      const newUrl = `http://localhost:${port}/v1/messages`
      return origFetch(newUrl, opts)
    }

    const task = {
      name: 'mock-gen',
      type: 'classification',
      labels: ['pos', 'neg', 'neutral'],
      description: 'test',
      synthetic: { count: 4, prompt: 'test', batchSize: 2, model: 'test-model' }
    }

    const progressCalls = []
    const result = await generate(task, {
      apiKey: 'test-key',
      onProgress: (cur, total) => progressCalls.push({ cur, total })
    })

    globalThis.fetch = origFetch

    expect(result.count).toBe(4)
    expect(result.path).toContain('mock-gen_synthetic.jsonl')

    // Verify written JSONL
    const data = await readJsonl(result.path)
    expect(data).toHaveLength(4)
    expect(data[0].text).toBe('great')

    // Verify progress was called
    expect(progressCalls.length).toBe(2)
    expect(progressCalls[0]).toEqual({ cur: 2, total: 4 })
    expect(progressCalls[1]).toEqual({ cur: 4, total: 4 })
  })

  test('handles API error', async () => {
    server?.stop()
    server = Bun.serve({
      port: 0,
      fetch() {
        return new Response('rate limited', { status: 429 })
      }
    })
    port = server.port

    const origFetch = globalThis.fetch
    globalThis.fetch = (url, opts) => origFetch(`http://localhost:${port}/v1/messages`, opts)

    const task = {
      name: 'mock-err',
      type: 'classification',
      labels: ['a', 'b'],
      description: 'test',
      synthetic: { count: 2, prompt: 'test', batchSize: 2 }
    }

    try {
      await generate(task, { apiKey: 'key', onProgress: () => {} })
      expect(true).toBe(false) // should not reach
    } catch (e) {
      expect(e.message).toContain('429')
    }

    globalThis.fetch = origFetch
  })
})
