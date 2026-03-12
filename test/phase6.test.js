import { test, expect, describe, afterAll } from 'bun:test'
import { callProvider, resolveProvider, listProviders, PROVIDERS, backoffMs } from '../lib/provider.js'
import { listTemplates, loadTemplate } from '../lib/templates.js'
import { generateReport, confusionMatrix, perLabelMetrics, findErrors } from '../lib/report.js'
import { join } from 'node:path'
import { rm } from 'node:fs/promises'

const ROOT = join(import.meta.dir, '..')
const REPORTS = join(ROOT, 'reports')

afterAll(async () => {
  await rm(join(REPORTS, 'test-report_report.html'), { force: true }).catch(() => {})
})

// ── Provider ─────────────────────────────────────────────

describe('provider', () => {
  test('PROVIDERS has anthropic, openai, ollama', () => {
    expect(PROVIDERS.anthropic).toBeDefined()
    expect(PROVIDERS.openai).toBeDefined()
    expect(PROVIDERS.ollama).toBeDefined()
  })

  test('listProviders returns all providers with config status', () => {
    const providers = listProviders()
    expect(providers.length).toBe(3)
    const keys = providers.map(p => p.key)
    expect(keys).toContain('anthropic')
    expect(keys).toContain('openai')
    expect(keys).toContain('ollama')
    // ollama has no envKey, so always configured
    const ollama = providers.find(p => p.key === 'ollama')
    expect(ollama.configured).toBe(true)
  })

  test('resolveProvider defaults to anthropic', () => {
    const p = resolveProvider({ synthetic: {} })
    expect(p.key).toBe('anthropic')
    expect(p.defaultModel).toBe('claude-sonnet-4-20250514')
  })

  test('resolveProvider picks openai from task config', () => {
    const p = resolveProvider({ synthetic: { provider: 'openai', model: 'gpt-4o' } })
    expect(p.key).toBe('openai')
    expect(p.model).toBe('gpt-4o')
  })

  test('resolveProvider picks ollama from task config', () => {
    const p = resolveProvider({ synthetic: { provider: 'ollama', model: 'mistral' } })
    expect(p.key).toBe('ollama')
    expect(p.model).toBe('mistral')
    expect(p.apiKey).toBeNull()
  })

  test('resolveProvider throws on unknown provider', () => {
    expect(() => resolveProvider({ synthetic: { provider: 'xyz' } })).toThrow('Unknown provider')
  })

  test('callProvider throws on unknown provider', async () => {
    try {
      await callProvider('xyz', { systemPrompt: 'test', userPrompt: 'test' })
      expect(true).toBe(false)
    } catch (e) {
      expect(e.message).toContain('Unknown provider')
    }
  })

  test('backoffMs increases with attempt', () => {
    const s0 = Array.from({ length: 50 }, () => backoffMs(0)).reduce((a, b) => a + b) / 50
    const s2 = Array.from({ length: 50 }, () => backoffMs(2)).reduce((a, b) => a + b) / 50
    expect(s2).toBeGreaterThan(s0)
  })

  test('callProvider with mock anthropic server', async () => {
    const server = Bun.serve({
      port: 0,
      fetch() {
        return new Response(JSON.stringify({
          content: [{ text: 'hello from mock' }]
        }))
      }
    })

    const text = await callProvider('anthropic', {
      apiKey: 'test-key',
      model: 'test-model',
      systemPrompt: 'system',
      userPrompt: 'user',
      url: `http://localhost:${server.port}`
    })
    expect(text).toBe('hello from mock')
    server.stop()
  })

  test('callProvider with mock openai server', async () => {
    const server = Bun.serve({
      port: 0,
      fetch() {
        return new Response(JSON.stringify({
          choices: [{ message: { content: 'hello from openai mock' } }]
        }))
      }
    })

    const text = await callProvider('openai', {
      apiKey: 'test-key',
      model: 'gpt-4o-mini',
      systemPrompt: 'system',
      userPrompt: 'user',
      url: `http://localhost:${server.port}`
    })
    expect(text).toBe('hello from openai mock')
    server.stop()
  })

  test('callProvider with mock ollama server', async () => {
    const server = Bun.serve({
      port: 0,
      fetch() {
        return new Response(JSON.stringify({
          message: { content: 'hello from ollama mock' }
        }))
      }
    })

    const text = await callProvider('ollama', {
      model: 'llama3',
      systemPrompt: 'system',
      userPrompt: 'user',
      url: `http://localhost:${server.port}`
    })
    expect(text).toBe('hello from ollama mock')
    server.stop()
  })
})

// ── Templates ────────────────────────────────────────────

describe('templates', () => {
  test('lists available templates', async () => {
    const templates = await listTemplates()
    expect(templates.length).toBeGreaterThanOrEqual(5)
    const names = templates.map(t => t.name)
    expect(names).toContain('sentiment')
    expect(names).toContain('intent')
    expect(names).toContain('spam-detector')
    expect(names).toContain('contact-extractor')
    expect(names).toContain('topic')
  })

  test('loads a specific template', async () => {
    const t = await loadTemplate('sentiment')
    expect(t.name).toBe('sentiment')
    expect(t.type).toBe('classification')
    expect(t.labels).toContain('positive')
    expect(t.synthetic.count).toBe(200)
  })

  test('loads template with .json extension', async () => {
    const t = await loadTemplate('intent.json')
    expect(t.name).toBe('intent')
  })

  test('throws on missing template', async () => {
    try {
      await loadTemplate('nonexistent')
      expect(true).toBe(false)
    } catch (e) {
      expect(e.message).toContain('not found')
    }
  })
})

// ── Report ───────────────────────────────────────────────

describe('report', () => {
  const labels = ['pos', 'neg']
  const actual = ['pos', 'pos', 'neg', 'neg', 'pos', 'neg']
  const predicted = ['pos', 'neg', 'neg', 'neg', 'pos', 'pos']

  test('confusionMatrix counts correctly', () => {
    const cm = confusionMatrix(actual, predicted, labels)
    expect(cm.pos.pos).toBe(2) // true positive
    expect(cm.pos.neg).toBe(1) // false negative
    expect(cm.neg.neg).toBe(2) // true negative
    expect(cm.neg.pos).toBe(1) // false positive
  })

  test('perLabelMetrics computes precision/recall/f1', () => {
    const m = perLabelMetrics(actual, predicted, labels)
    // pos: tp=2, fp=1, fn=1 → precision=2/3, recall=2/3, f1=2/3
    expect(m.pos.precision).toBeCloseTo(2 / 3, 2)
    expect(m.pos.recall).toBeCloseTo(2 / 3, 2)
    expect(m.pos.f1).toBeCloseTo(2 / 3, 2)
    // neg: tp=2, fp=1, fn=1
    expect(m.neg.precision).toBeCloseTo(2 / 3, 2)
  })

  test('findErrors returns misclassified examples', () => {
    const data = [
      { text: 'good', label: 'pos' },
      { text: 'bad call', label: 'pos' },
      { text: 'awful', label: 'neg' },
      { text: 'meh', label: 'neg' },
      { text: 'great', label: 'pos' },
      { text: 'wrong', label: 'neg' },
    ]
    const preds = ['pos', 'neg', 'neg', 'neg', 'pos', 'pos']
    const errors = findErrors(data, preds)
    expect(errors['pos → neg']).toHaveLength(1)
    expect(errors['pos → neg'][0].text).toBe('bad call')
    expect(errors['neg → pos']).toHaveLength(1)
    expect(errors['neg → pos'][0].text).toBe('wrong')
  })

  test('generateReport creates HTML file', async () => {
    const valData = [
      { text: 'love it', label: 'pos' },
      { text: 'hate it', label: 'neg' },
      { text: 'okay', label: 'pos' },
      { text: 'bad', label: 'neg' },
    ]
    const predictions = [
      { label: 'pos' },
      { label: 'neg' },
      { label: 'neg' },
      { label: 'neg' },
    ]
    const meta = { accuracy: 0.75, algorithm: 'logistic_regression', train_size: 100, val_size: 4, labels: ['pos', 'neg'] }
    const result = await generateReport('test-report', { valData, predictions, labels: ['pos', 'neg'], meta })

    expect(result.path).toContain('test-report_report.html')
    expect(result.accuracy).toBe(0.75)

    const html = await Bun.file(result.path).text()
    expect(html).toContain('test-report')
    expect(html).toContain('Confusion Matrix')
    expect(html).toContain('Per-Label Metrics')
    expect(html).toContain('Example Errors')
  })
})
