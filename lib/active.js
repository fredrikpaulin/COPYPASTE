import { join } from 'node:path'

const SCRIPTS_DIR = join(import.meta.dir, '..', 'scripts')
const MODELS_DIR = join(import.meta.dir, '..', 'models')

// Get prediction confidence for a batch of texts using the trained model
async function getUncertainExamples(taskName, texts, { topK = 10 } = {}) {
  const { spawn } = await import('node:child_process')
  const scriptPath = join(SCRIPTS_DIR, 'train.py')
  const modelPath = join(MODELS_DIR, taskName, 'model.pkl')

  return new Promise((resolve, reject) => {
    const proc = spawn('python3', [scriptPath, '--predict', modelPath, '--input', '-'], {
      cwd: join(import.meta.dir, '..'),
      env: { ...process.env }
    })

    for (const text of texts) {
      proc.stdin.write(JSON.stringify({ text }) + '\n')
    }
    proc.stdin.end()

    let stdout = ''
    let stderr = ''
    proc.stdout.on('data', chunk => { stdout += chunk.toString() })
    proc.stderr.on('data', chunk => { stderr += chunk.toString() })

    proc.on('close', code => {
      if (code !== 0) return reject(new Error(`Prediction failed: ${stderr}`))
      try {
        const predictions = JSON.parse(stdout)
        // Sort by confidence ascending (least confident first)
        const sorted = predictions
          .filter(p => p.confidence !== null)
          .sort((a, b) => a.confidence - b.confidence)
          .slice(0, topK)
        resolve(sorted)
      } catch (e) {
        reject(new Error(`Failed to parse predictions: ${e.message}`))
      }
    })

    proc.on('error', err => reject(new Error(`Failed to spawn python3: ${err.message}`)))
  })
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
async function llmLabel(examples, task, { apiKey, model }) {
  const prompt = `You are labeling training examples for a ${task.type} task: "${task.description}".

For each example below, assign the correct label from: ${(task.labels || []).join(', ')}

Respond with a JSON array of objects: {"text": "...", "label": "correct_label"}

Examples to label:
${examples.map((e, i) => `${i + 1}. "${e.text}" (model predicted: ${e.predicted}, confidence: ${(e.confidence * 100).toFixed(0)}%)`).join('\n')}

Respond with ONLY the JSON array.`

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01'
    },
    body: JSON.stringify({
      model: model || 'claude-sonnet-4-20250514',
      max_tokens: 4096,
      messages: [{ role: 'user', content: prompt }]
    })
  })

  if (!res.ok) {
    throw new Error(`Claude API ${res.status}: ${await res.text()}`)
  }

  const data = await res.json()
  const text = data.content[0].text
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
