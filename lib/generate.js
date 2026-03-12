import { join } from 'node:path'
import { callProvider, resolveProvider } from './provider.js'

const DATA_DIR = join(import.meta.dir, '..', 'data')

// Re-export backoffMs from provider for backward compat
export { backoffMs } from './provider.js'

// Legacy wrapper — calls Anthropic directly (used by tests that mock fetch)
function backoffMs_local(attempt, base = 1000, max = 60000) {
  const exp = Math.min(base * 2 ** attempt, max)
  return exp / 2 + Math.random() * (exp / 2)
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)) }

async function callClaude(apiKey, model, systemPrompt, userPrompt, { maxRetries = 3, onRetry } = {}) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model,
        max_tokens: 4096,
        system: systemPrompt,
        messages: [{ role: 'user', content: userPrompt }]
      })
    })

    if (res.ok) {
      const data = await res.json()
      return data.content[0].text
    }

    const status = res.status
    const body = await res.text()

    if (status !== 429 && status !== 529 && status < 500) {
      throw new Error(`Claude API ${status}: ${body}`)
    }

    if (attempt === maxRetries) {
      throw new Error(`Claude API ${status} after ${maxRetries + 1} attempts: ${body}`)
    }

    const retryAfter = res.headers.get('retry-after')
    const waitMs = retryAfter
      ? parseInt(retryAfter) * 1000
      : backoffMs_local(attempt)

    onRetry?.({ attempt: attempt + 1, maxRetries, waitMs, status })
    await sleep(waitMs)
  }
}

function buildSystemPrompt(task) {
  const base = `You are a training data generator. Your job is to produce realistic, diverse examples for a machine learning task.

Task: ${task.description}
Type: ${task.type}`

  if (task.type === 'classification') {
    return base + `\nLabels: ${task.labels.join(', ')}`
  }
  if (task.type === 'extraction') {
    return base + `\nFields to extract: ${task.fields.join(', ')}`
  }
  return base
}

function buildBatchPrompt(task, batchSize) {
  const format = task.type === 'classification'
    ? '{"text": "...", "label": "..."}'
    : task.type === 'extraction'
    ? '{"text": "...", "fields": {...}}'
    : '{"text": "...", "value": ...}'

  let labelInstruction = ''
  if (task.type === 'classification') {
    labelInstruction = `\nDistribute labels roughly evenly across: ${task.labels.join(', ')}`
  }

  const userPrompt = task.synthetic.prompt
    .replace('{label}', task.labels ? task.labels.join(' or ') : '')
    .replace('{field}', task.fields ? task.fields.join(', ') : '')

  return `Generate exactly ${batchSize} training examples as a JSON array.
Each example should follow this format: ${format}
${labelInstruction}

Context: ${userPrompt}

Respond with ONLY a valid JSON array, no other text.`
}

function parseBatchResponse(text) {
  const match = text.match(/\[[\s\S]*\]/)
  if (!match) throw new Error('No JSON array found in response')
  return JSON.parse(match[0])
}

function validateExample(example, task) {
  if (!example || typeof example !== 'object') return false
  if (typeof example.text !== 'string' || !example.text.trim()) return false

  if (task.type === 'classification') {
    if (typeof example.label !== 'string') return false
    if (task.labels && !task.labels.includes(example.label)) return false
  }
  if (task.type === 'extraction') {
    if (!example.fields || typeof example.fields !== 'object') return false
  }
  if (task.type === 'regression') {
    if (typeof example.value !== 'number') return false
  }
  return true
}

async function generate(task, { apiKey, onProgress, onRetry, onDropped, provider: providerOverride }) {
  const providerName = providerOverride || task.synthetic?.provider || 'anthropic'
  const model = task.synthetic?.model
  const total = task.synthetic?.count || 100
  const batchSize = task.synthetic?.batchSize || 10
  const batches = Math.ceil(total / batchSize)
  const systemPrompt = buildSystemPrompt(task)

  const allExamples = []
  let droppedCount = 0

  for (let i = 0; i < batches; i++) {
    const remaining = total - allExamples.length
    const thisBatch = Math.min(batchSize, remaining)
    const userPrompt = buildBatchPrompt(task, thisBatch)

    let text
    if (providerName === 'anthropic') {
      // Use legacy callClaude for backward compat (tests mock globalThis.fetch)
      text = await callClaude(apiKey, model || 'claude-sonnet-4-20250514', systemPrompt, userPrompt, {
        maxRetries: 3,
        onRetry
      })
    } else {
      text = await callProvider(providerName, {
        apiKey,
        model,
        systemPrompt,
        userPrompt,
        url: task.synthetic?.url,
        maxRetries: 3,
        onRetry
      })
    }

    const raw = parseBatchResponse(text)

    for (const example of raw) {
      if (validateExample(example, task)) {
        allExamples.push(example)
      } else {
        droppedCount++
      }
    }

    onProgress?.(Math.min(allExamples.length, total), total)
  }

  if (droppedCount > 0) {
    onDropped?.(droppedCount)
  }

  const outPath = join(DATA_DIR, `${task.name}_synthetic.jsonl`)
  const lines = allExamples.slice(0, total).map(e => JSON.stringify(e)).join('\n') + '\n'
  await Bun.write(outPath, lines)

  return { count: Math.min(allExamples.length, total), path: outPath, dropped: droppedCount }
}

async function preview(task, { apiKey, count = 5, onRetry, provider: providerOverride }) {
  const providerName = providerOverride || task.synthetic?.provider || 'anthropic'
  const model = task.synthetic?.model
  const systemPrompt = buildSystemPrompt(task)
  const userPrompt = buildBatchPrompt(task, count)

  let text
  if (providerName === 'anthropic') {
    text = await callClaude(apiKey, model || 'claude-sonnet-4-20250514', systemPrompt, userPrompt, {
      maxRetries: 2,
      onRetry
    })
  } else {
    text = await callProvider(providerName, {
      apiKey,
      model,
      systemPrompt,
      userPrompt,
      url: task.synthetic?.url,
      maxRetries: 2,
      onRetry
    })
  }

  const raw = parseBatchResponse(text)
  const valid = raw.filter(e => validateExample(e, task))
  const dropped = raw.length - valid.length
  return { examples: valid, dropped }
}

export {
  generate, preview,
  buildSystemPrompt, buildBatchPrompt, parseBatchResponse, validateExample
}
