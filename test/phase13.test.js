import { test, expect, describe } from 'bun:test'
import {
  callProvider, streamProvider, resolveProvider, listProviders, PROVIDERS, backoffMs,
  parseSSE, parseNDJSON, extractAnthropicToken, extractOpenAIToken, extractOllamaToken
} from '../lib/provider.js'
import { streamBox } from '../lib/tui.js'

// ── SSE/NDJSON Parsing ─────────────────────────────────────

describe('parseSSE', () => {
  function makeReader(chunks) {
    let i = 0
    return {
      async read() {
        if (i >= chunks.length) return { done: true, value: undefined }
        return { done: false, value: new TextEncoder().encode(chunks[i++]) }
      }
    }
  }

  test('parses single SSE event', async () => {
    const reader = makeReader(['data: {"text":"hello"}\n\n'])
    const events = []
    for await (const ev of parseSSE(reader)) events.push(ev)
    expect(events).toEqual(['{"text":"hello"}'])
  })

  test('parses multiple SSE events', async () => {
    const reader = makeReader(['data: first\ndata: second\n\n'])
    const events = []
    for await (const ev of parseSSE(reader)) events.push(ev)
    expect(events).toEqual(['first', 'second'])
  })

  test('handles split across chunks', async () => {
    const reader = makeReader(['data: hel', 'lo\n\n'])
    const events = []
    for await (const ev of parseSSE(reader)) events.push(ev)
    expect(events).toEqual(['hello'])
  })

  test('stops at [DONE]', async () => {
    const reader = makeReader(['data: first\ndata: [DONE]\ndata: ignored\n\n'])
    const events = []
    for await (const ev of parseSSE(reader)) events.push(ev)
    expect(events).toEqual(['first'])
  })

  test('ignores non-data lines', async () => {
    const reader = makeReader(['event: message\ndata: hello\nid: 1\n\n'])
    const events = []
    for await (const ev of parseSSE(reader)) events.push(ev)
    expect(events).toEqual(['hello'])
  })
})

describe('parseNDJSON', () => {
  function makeReader(chunks) {
    let i = 0
    return {
      async read() {
        if (i >= chunks.length) return { done: true, value: undefined }
        return { done: false, value: new TextEncoder().encode(chunks[i++]) }
      }
    }
  }

  test('parses NDJSON lines', async () => {
    const reader = makeReader(['{"a":1}\n{"b":2}\n'])
    const lines = []
    for await (const line of parseNDJSON(reader)) lines.push(line)
    expect(lines).toEqual(['{"a":1}', '{"b":2}'])
  })

  test('handles split across chunks', async () => {
    const reader = makeReader(['{"a":', '1}\n'])
    const lines = []
    for await (const line of parseNDJSON(reader)) lines.push(line)
    expect(lines).toEqual(['{"a":1}'])
  })

  test('skips empty lines', async () => {
    const reader = makeReader(['\n{"a":1}\n\n'])
    const lines = []
    for await (const line of parseNDJSON(reader)) lines.push(line)
    expect(lines).toEqual(['{"a":1}'])
  })
})

// ── Token Extractors ──────────────────────────────────────

describe('extractAnthropicToken', () => {
  test('extracts text from content_block_delta', () => {
    const data = JSON.stringify({ type: 'content_block_delta', delta: { text: 'hi' } })
    expect(extractAnthropicToken(data)).toBe('hi')
  })

  test('returns null for other event types', () => {
    expect(extractAnthropicToken(JSON.stringify({ type: 'message_start' }))).toBeNull()
    expect(extractAnthropicToken(JSON.stringify({ type: 'content_block_start' }))).toBeNull()
    expect(extractAnthropicToken(JSON.stringify({ type: 'message_stop' }))).toBeNull()
  })

  test('returns null for invalid JSON', () => {
    expect(extractAnthropicToken('not json')).toBeNull()
  })
})

describe('extractOpenAIToken', () => {
  test('extracts content from delta', () => {
    const data = JSON.stringify({ choices: [{ delta: { content: 'world' } }] })
    expect(extractOpenAIToken(data)).toBe('world')
  })

  test('returns null when no content', () => {
    const data = JSON.stringify({ choices: [{ delta: {} }] })
    expect(extractOpenAIToken(data)).toBeNull()
  })

  test('returns null for invalid JSON', () => {
    expect(extractOpenAIToken('bad')).toBeNull()
  })
})

describe('extractOllamaToken', () => {
  test('extracts content from message', () => {
    const data = JSON.stringify({ message: { content: 'ok' }, done: false })
    expect(extractOllamaToken(data)).toBe('ok')
  })

  test('returns null when done', () => {
    const data = JSON.stringify({ message: { content: '' }, done: true })
    expect(extractOllamaToken(data)).toBeNull()
  })

  test('returns null for invalid JSON', () => {
    expect(extractOllamaToken('bad')).toBeNull()
  })
})

// ── streamProvider with mock servers ──────────────────────

describe('streamProvider', () => {
  test('throws on unknown provider', async () => {
    try {
      await streamProvider('xyz', { systemPrompt: 'a', userPrompt: 'b' })
      expect(true).toBe(false)
    } catch (e) {
      expect(e.message).toContain('Unknown provider')
    }
  })

  test('streams from mock anthropic SSE server', async () => {
    const sseBody = [
      'data: {"type":"content_block_delta","delta":{"text":"hel"}}\n\n',
      'data: {"type":"content_block_delta","delta":{"text":"lo"}}\n\n',
      'data: {"type":"message_stop"}\n\n',
      'data: [DONE]\n\n'
    ].join('')

    const server = Bun.serve({
      port: 0,
      fetch() {
        return new Response(sseBody, {
          headers: { 'Content-Type': 'text/event-stream' }
        })
      }
    })

    const tokens = []
    const result = await streamProvider('anthropic', {
      apiKey: 'test',
      model: 'test',
      systemPrompt: 'sys',
      userPrompt: 'usr',
      url: `http://localhost:${server.port}`,
      onToken: (token) => tokens.push(token)
    })

    expect(result).toBe('hello')
    expect(tokens).toEqual(['hel', 'lo'])
    server.stop()
  })

  test('streams from mock openai SSE server', async () => {
    const sseBody = [
      'data: {"choices":[{"delta":{"content":"wor"}}]}\n\n',
      'data: {"choices":[{"delta":{"content":"ld"}}]}\n\n',
      'data: [DONE]\n\n'
    ].join('')

    const server = Bun.serve({
      port: 0,
      fetch() {
        return new Response(sseBody, {
          headers: { 'Content-Type': 'text/event-stream' }
        })
      }
    })

    const tokens = []
    const result = await streamProvider('openai', {
      apiKey: 'test',
      model: 'test',
      systemPrompt: 'sys',
      userPrompt: 'usr',
      url: `http://localhost:${server.port}`,
      onToken: (token) => tokens.push(token)
    })

    expect(result).toBe('world')
    expect(tokens).toEqual(['wor', 'ld'])
    server.stop()
  })

  test('streams from mock ollama NDJSON server', async () => {
    const ndjson = [
      '{"message":{"content":"foo"},"done":false}',
      '{"message":{"content":"bar"},"done":false}',
      '{"message":{"content":""},"done":true}'
    ].join('\n') + '\n'

    const server = Bun.serve({
      port: 0,
      fetch() {
        return new Response(ndjson, {
          headers: { 'Content-Type': 'application/x-ndjson' }
        })
      }
    })

    const tokens = []
    const result = await streamProvider('ollama', {
      model: 'llama3',
      systemPrompt: 'sys',
      userPrompt: 'usr',
      url: `http://localhost:${server.port}`,
      onToken: (token) => tokens.push(token)
    })

    expect(result).toBe('foobar')
    expect(tokens).toEqual(['foo', 'bar'])
    server.stop()
  })

  test('onToken receives accumulated text', async () => {
    const sseBody = [
      'data: {"type":"content_block_delta","delta":{"text":"a"}}\n\n',
      'data: {"type":"content_block_delta","delta":{"text":"b"}}\n\n',
      'data: {"type":"content_block_delta","delta":{"text":"c"}}\n\n',
      'data: [DONE]\n\n'
    ].join('')

    const server = Bun.serve({
      port: 0,
      fetch() {
        return new Response(sseBody, {
          headers: { 'Content-Type': 'text/event-stream' }
        })
      }
    })

    const accumulated = []
    await streamProvider('anthropic', {
      apiKey: 'test',
      model: 'test',
      systemPrompt: 'sys',
      userPrompt: 'usr',
      url: `http://localhost:${server.port}`,
      onToken: (token, full) => accumulated.push(full)
    })

    expect(accumulated).toEqual(['a', 'ab', 'abc'])
    server.stop()
  })

  test('handles HTTP error with retry', async () => {
    let attempts = 0
    const server = Bun.serve({
      port: 0,
      fetch() {
        attempts++
        if (attempts <= 2) {
          return new Response('rate limited', { status: 429 })
        }
        return new Response(
          'data: {"type":"content_block_delta","delta":{"text":"ok"}}\n\ndata: [DONE]\n\n',
          { headers: { 'Content-Type': 'text/event-stream' } }
        )
      }
    })

    const retries = []
    const result = await streamProvider('anthropic', {
      apiKey: 'test',
      model: 'test',
      systemPrompt: 'sys',
      userPrompt: 'usr',
      url: `http://localhost:${server.port}`,
      maxRetries: 3,
      onRetry: (info) => retries.push(info.attempt)
    })

    expect(result).toBe('ok')
    expect(retries).toEqual([1, 2])
    server.stop()
  })

  test('throws on non-retryable error', async () => {
    const server = Bun.serve({
      port: 0,
      fetch() {
        return new Response('bad request', { status: 400 })
      }
    })

    try {
      await streamProvider('anthropic', {
        apiKey: 'test',
        model: 'test',
        systemPrompt: 'sys',
        userPrompt: 'usr',
        url: `http://localhost:${server.port}`
      })
      expect(true).toBe(false)
    } catch (e) {
      expect(e.message).toContain('400')
    }
    server.stop()
  })
})

// ── streamBox TUI helper ─────────────────────────────────

describe('streamBox', () => {
  test('exports streamBox function', () => {
    expect(typeof streamBox).toBe('function')
  })

  test('returns object with write, end, chars', () => {
    // Capture stdout
    const orig = process.stdout.write
    let output = ''
    process.stdout.write = (s) => { output += s; return true }

    const box = streamBox('Test')
    expect(typeof box.write).toBe('function')
    expect(typeof box.end).toBe('function')
    expect(typeof box.chars).toBe('function')

    box.write('hello')
    expect(box.chars()).toBe(5)
    box.end()

    process.stdout.write = orig
  })

  test('tracks character count', () => {
    const orig = process.stdout.write
    process.stdout.write = () => true

    const box = streamBox('Test')
    box.write('abc')
    box.write('de')
    expect(box.chars()).toBe(5)
    box.end()

    process.stdout.write = orig
  })
})

// ── generate.js streaming support ─────────────────────────

describe('generate streaming', () => {
  test('generate accepts stream and onToken options', async () => {
    const { generate } = await import('../lib/generate.js')

    // Mock server that returns valid JSON array via SSE
    const jsonArray = JSON.stringify([
      { text: 'streaming test 1', label: 'positive' },
      { text: 'streaming test 2', label: 'negative' }
    ])
    const sseBody = `data: {"type":"content_block_delta","delta":{"text":"${jsonArray.replace(/"/g, '\\"')}"}}\n\ndata: [DONE]\n\n`

    const server = Bun.serve({
      port: 0,
      fetch() {
        return new Response(sseBody, {
          headers: { 'Content-Type': 'text/event-stream' }
        })
      }
    })

    const task = {
      name: 'test-stream',
      type: 'classification',
      description: 'test',
      labels: ['positive', 'negative'],
      synthetic: {
        prompt: 'test {label}',
        count: 2,
        batchSize: 5,
        provider: 'anthropic',
        url: `http://localhost:${server.port}`
      }
    }

    const tokens = []
    const result = await generate(task, {
      apiKey: 'test',
      stream: true,
      onToken: (token) => tokens.push(token),
      onProgress: () => {}
    })

    expect(tokens.length).toBeGreaterThan(0)
    expect(result.count).toBe(2)
    server.stop()
  })

  test('preview accepts stream and onToken options', async () => {
    const { preview } = await import('../lib/generate.js')

    const jsonArray = JSON.stringify([
      { text: 'stream preview', label: 'positive' }
    ])
    const sseBody = `data: {"type":"content_block_delta","delta":{"text":"${jsonArray.replace(/"/g, '\\"')}"}}\n\ndata: [DONE]\n\n`

    const server = Bun.serve({
      port: 0,
      fetch() {
        return new Response(sseBody, {
          headers: { 'Content-Type': 'text/event-stream' }
        })
      }
    })

    const task = {
      name: 'test-stream-preview',
      type: 'classification',
      description: 'test',
      labels: ['positive', 'negative'],
      synthetic: {
        prompt: 'test {label}',
        count: 1,
        batchSize: 5,
        provider: 'anthropic',
        url: `http://localhost:${server.port}`
      }
    }

    const tokens = []
    const result = await preview(task, {
      apiKey: 'test',
      stream: true,
      onToken: (token) => tokens.push(token)
    })

    expect(tokens.length).toBeGreaterThan(0)
    expect(result.examples.length).toBe(1)
    server.stop()
  })
})

// ── Export consistency ─────────────────────────────────────

describe('exports', () => {
  test('provider.js exports streamProvider', () => {
    expect(typeof streamProvider).toBe('function')
  })

  test('provider.js exports all parser functions', () => {
    expect(typeof parseSSE).toBe('function')
    expect(typeof parseNDJSON).toBe('function')
    expect(typeof extractAnthropicToken).toBe('function')
    expect(typeof extractOpenAIToken).toBe('function')
    expect(typeof extractOllamaToken).toBe('function')
  })

  test('provider.js still exports batch functions', () => {
    expect(typeof callProvider).toBe('function')
    expect(typeof resolveProvider).toBe('function')
    expect(typeof listProviders).toBe('function')
  })
})
