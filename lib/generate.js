import { join } from 'node:path'

const DATA_DIR = join(import.meta.dir, '..', 'data')

// Call Claude API directly — no SDK
async function callClaude(apiKey, model, systemPrompt, userPrompt) {
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
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Claude API ${res.status}: ${body}`)
  }
  const data = await res.json()
  return data.content[0].text
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
  // Extract JSON array from response, handling potential markdown fences
  const match = text.match(/\[[\s\S]*\]/)
  if (!match) throw new Error('No JSON array found in response')
  return JSON.parse(match[0])
}

async function generate(task, { apiKey, onProgress }) {
  const model = task.synthetic?.model || 'claude-sonnet-4-20250514'
  const total = task.synthetic?.count || 100
  const batchSize = task.synthetic?.batchSize || 10
  const batches = Math.ceil(total / batchSize)
  const systemPrompt = buildSystemPrompt(task)

  const allExamples = []

  for (let i = 0; i < batches; i++) {
    const remaining = total - allExamples.length
    const thisBatch = Math.min(batchSize, remaining)
    const userPrompt = buildBatchPrompt(task, thisBatch)

    const text = await callClaude(apiKey, model, systemPrompt, userPrompt)
    const examples = parseBatchResponse(text)
    allExamples.push(...examples)

    onProgress?.(allExamples.length, total)
  }

  // Write to JSONL
  const outPath = join(DATA_DIR, `${task.name}_synthetic.jsonl`)
  const lines = allExamples.slice(0, total).map(e => JSON.stringify(e)).join('\n') + '\n'
  await Bun.write(outPath, lines)

  return { count: Math.min(allExamples.length, total), path: outPath }
}

export { generate, buildSystemPrompt, buildBatchPrompt, parseBatchResponse }
