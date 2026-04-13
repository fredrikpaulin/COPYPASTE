import { join } from 'node:path'
import { callProvider, streamProvider, resolveProvider } from './provider.js'

const DATA_DIR = join(import.meta.dir, '..', 'data')

// Re-export backoffMs from provider for backward compat
export { backoffMs } from './provider.js'

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
  if (task.type === 'sequence-labeling') {
    return base + `\nEntity types: ${task.labels.join(', ')}\nTag scheme: BIO (B-TYPE for first token, I-TYPE for continuation, O for outside)`
  }
  if (task.type === 'scoring' || task.type === 'regression') {
    const range = task.scoreRange || { min: 0, max: 5 }
    return base + `\nScore range: ${range.min} to ${range.max} (numeric, decimals allowed)`
  }
  return base
}

function buildBatchPrompt(task, batchSize) {
  if (task.type === 'sequence-labeling') {
    const userPrompt = task.synthetic.prompt
      .replace('{label}', task.labels ? task.labels.join(' or ') : '')
    return `Generate exactly ${batchSize} training examples as a JSON array.
Each example should follow this format: {"tokens": ["word1", "word2", ...], "tags": ["O", "B-TYPE", "I-TYPE", ...]}

Use BIO tagging with these entity types: ${task.labels.join(', ')}
- B-TYPE marks the first token of an entity
- I-TYPE marks continuation tokens
- O marks tokens outside any entity

Include a mix of sentences: some with multiple entities, some with one, some with none.
Make the text natural and varied in style and length.

Context: ${userPrompt}

Respond with ONLY a valid JSON array, no other text.`
  }

  const format = task.type === 'classification'
    ? '{"text": "...", "label": "..."}'
    : task.type === 'extraction'
    ? '{"text": "...", "fields": {...}}'
    : '{"text": "...", "value": <number>}'

  let labelInstruction = ''
  if (task.type === 'classification') {
    labelInstruction = `\nDistribute labels roughly evenly across: ${task.labels.join(', ')}`
  }
  if (task.type === 'scoring' || task.type === 'regression') {
    const range = task.scoreRange || { min: 0, max: 5 }
    labelInstruction = `\nEach "value" must be a number between ${range.min} and ${range.max}. Distribute scores across the full range — include low, mid, and high scores. Decimals are encouraged for variety.`
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

  if (task.type === 'sequence-labeling') {
    if (!Array.isArray(example.tokens) || !example.tokens.length) return false
    if (!Array.isArray(example.tags) || example.tags.length !== example.tokens.length) return false
    // Every tag must be O, B-<label>, or I-<label>
    const validTags = new Set(['O'])
    for (const l of task.labels || []) { validTags.add(`B-${l}`); validTags.add(`I-${l}`) }
    for (const t of example.tags) {
      if (!validTags.has(t)) return false
    }
    return true
  }

  if (typeof example.text !== 'string' || !example.text.trim()) return false

  if (task.type === 'classification') {
    if (typeof example.label !== 'string') return false
    if (task.labels && !task.labels.includes(example.label)) return false
  }
  if (task.type === 'extraction') {
    if (!example.fields || typeof example.fields !== 'object') return false
  }
  if (task.type === 'regression' || task.type === 'scoring') {
    if (typeof example.value !== 'number') return false
    if (isNaN(example.value)) return false
    if (task.scoreRange) {
      if (example.value < task.scoreRange.min || example.value > task.scoreRange.max) return false
    }
  }
  return true
}

async function generate(task, { apiKey, onProgress, onRetry, onDropped, onToken, stream = false, provider: providerOverride }) {
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
    if (stream) {
      text = await streamProvider(providerName, {
        apiKey,
        model: model || (providerName === 'anthropic' ? 'claude-sonnet-4-20250514' : undefined),
        systemPrompt,
        userPrompt,
        url: task.synthetic?.url,
        maxRetries: 3,
        onRetry,
        onToken: (token, full) => onToken?.(token, full, { batch: i + 1, batches })
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

async function preview(task, { apiKey, count = 5, onRetry, onToken, stream = false, provider: providerOverride }) {
  const providerName = providerOverride || task.synthetic?.provider || 'anthropic'
  const model = task.synthetic?.model
  const systemPrompt = buildSystemPrompt(task)
  const userPrompt = buildBatchPrompt(task, count)

  let text
  if (stream) {
    text = await streamProvider(providerName, {
      apiKey,
      model: model || (providerName === 'anthropic' ? 'claude-sonnet-4-20250514' : undefined),
      systemPrompt,
      userPrompt,
      url: task.synthetic?.url,
      maxRetries: 2,
      onRetry,
      onToken
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
