import { join } from 'node:path'

const SCRIPTS_DIR = join(import.meta.dir, '..', 'scripts')
const MODELS_DIR = join(import.meta.dir, '..', 'models')

// Get prediction confidence for a batch of texts using the trained model
async function getUncertainExamples(taskName, texts, { topK = 10 } = {}) {
  const scriptPath = join(SCRIPTS_DIR, 'train.py')
  const modelPath = join(MODELS_DIR, taskName, 'model.pkl')

  const proc = Bun.spawn(['python3', scriptPath, '--predict', modelPath, '--input', '-'], {
    cwd: join(import.meta.dir, '..'),
    stdin: 'pipe',
    stdout: 'pipe',
    stderr: 'ignore'
  })

  for (const text of texts) {
    proc.stdin.write(JSON.stringify({ text }) + '\n')
  }
  proc.stdin.end()

  const stdout = await new Response(proc.stdout).text()
  const code = await proc.exited
  if (code !== 0) throw new Error(`Prediction failed (exit code ${code})`)

  try {
    const predictions = JSON.parse(stdout)
    return predictions
      .filter(p => p.confidence !== null)
      .sort((a, b) => a.confidence - b.confidence)
      .slice(0, topK)
  } catch (e) {
    throw new Error(`Failed to parse predictions: ${e.message}`)
  }
}

// Generate candidate examples via Claude, then rank them by model uncertainty
async function generateAndRankByUncertainty(task, { apiKey, count = 20, topK = 10, onProgress }) {
  // Generate candidates
  const { generate, preview } = await import('./generate.js')
  const candidates = await preview(task, { apiKey, count })

  if (!candidates.examples.length) {
    return { uncertain: [], generated: 0 }
  }

  const texts = candidates.examples.map(e => e.text)
  const uncertain = await getUncertainExamples(task.name, texts, { topK })

  // Match uncertain predictions back to full examples
  const textToExample = new Map(candidates.examples.map(e => [e.text, e]))
  const results = uncertain.map(u => ({
    ...textToExample.get(u.text),
    predicted: u.label,
    confidence: u.confidence
  }))

  onProgress?.(results.length, count)
  return { uncertain: results, generated: candidates.examples.length }
}

// LLM-in-the-loop: send uncertain examples to Claude for labeling
async function llmLabel(examples, task, { apiKey, model, provider = 'anthropic' }) {
  const { callProvider } = await import('./provider.js')

  const systemPrompt = `You are labeling training examples for a ${task.type} task: "${task.description}".`
  const userPrompt = `For each example below, assign the correct label from: ${(task.labels || []).join(', ')}

Respond with a JSON array of objects: {"text": "...", "label": "correct_label"}

Examples to label:
${examples.map((e, i) => `${i + 1}. "${e.text}" (model predicted: ${e.predicted}, confidence: ${(e.confidence * 100).toFixed(0)}%)`).join('\n')}

Respond with ONLY the JSON array.`

  const text = await callProvider(provider, {
    apiKey,
    model,
    systemPrompt,
    userPrompt,
    maxRetries: 2
  })

  const match = text.match(/\[[\s\S]*\]/)
  if (!match) throw new Error('No JSON array found in labeling response')
  return JSON.parse(match[0])
}

// Track iteration history
async function loadHistory(taskName) {
  const historyPath = join(MODELS_DIR, taskName, 'active_history.json')
  const file = Bun.file(historyPath)
  if (await file.exists()) return file.json()
  return { iterations: [] }
}

async function saveIteration(taskName, iteration) {
  const historyPath = join(MODELS_DIR, taskName, 'active_history.json')
  const history = await loadHistory(taskName)
  history.iterations.push({
    ...iteration,
    timestamp: new Date().toISOString()
  })
  await Bun.write(historyPath, JSON.stringify(history, null, 2) + '\n')
  return history
}

export {
  getUncertainExamples,
  generateAndRankByUncertainty,
  llmLabel,
  loadHistory,
  saveIteration
}
