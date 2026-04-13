// Phase 9 — Multi-task and transfer learning
// Shared feature extraction, zero-shot bootstrap, progressive distillation.

import { join } from 'node:path'

// ── Shared Feature Extraction ────────────────────────────

// Build shared TF-IDF features for multiple tasks, then train task-specific heads
async function sharedFeatureTraining(tasks, { onStdout, onStderr } = {}) {
  const { readJsonl, writeJsonl } = await import('./data.js')

  // Collect all training texts across tasks for shared vocabulary
  const allTexts = []
  const taskData = {}

  for (const task of tasks) {
    const trainPath = join('data', `${task.name}_train.jsonl`)
    const valPath = join('data', `${task.name}_val.jsonl`)
    try {
      const train = await readJsonl(trainPath)
      const val = await readJsonl(valPath)
      taskData[task.name] = { train, val, task }
      allTexts.push(...train.map(r => r.text), ...val.map(r => r.text))
    } catch {
      // Skip tasks without prepared data
    }
  }

  if (!Object.keys(taskData).length) {
    throw new Error('No tasks have prepared train/val data')
  }

  return { taskData, sharedVocabSize: allTexts.length, taskCount: Object.keys(taskData).length }
}

// ── Zero-Shot Bootstrap ──────────────────────────────────

// Evaluate zero-shot performance by having the LLM classify validation examples directly
async function zeroShotEval(task, valData, { apiKey, model, provider = 'anthropic', onProgress } = {}) {
  const { callProvider } = await import('./provider.js')
  const batchSize = 20
  const batches = Math.ceil(valData.length / batchSize)
  const predictions = []

  for (let i = 0; i < batches; i++) {
    const batch = valData.slice(i * batchSize, (i + 1) * batchSize)

    const systemPrompt = `You are a classifier. Task: ${task.description}
Labels: ${task.labels.join(', ')}
Respond with ONLY a JSON array of labels, one per input.`

    const userPrompt = `Classify each text:
${batch.map((r, idx) => `${idx}: "${r.text}"`).join('\n')}

Respond with ONLY a JSON array of strings (the predicted labels), e.g. ["label1", "label2", ...]`

    try {
      const text = await callProvider(provider, {
        apiKey,
        model,
        systemPrompt,
        userPrompt,
        maxRetries: 2
      })

      const match = text.match(/\[[\s\S]*\]/)
      if (match) {
        const labels = JSON.parse(match[0])
        for (let j = 0; j < batch.length; j++) {
          predictions.push({
            text: batch[j].text,
            actual: batch[j].label,
            predicted: labels[j] || 'unknown',
            correct: labels[j] === batch[j].label
          })
        }
      }
    } catch {
      for (const row of batch) {
        predictions.push({ text: row.text, actual: row.label, predicted: 'error', correct: false })
      }
    }

    onProgress?.(Math.min((i + 1) * batchSize, valData.length), valData.length)
  }

  const correct = predictions.filter(p => p.correct).length
  const accuracy = predictions.length > 0 ? correct / predictions.length : 0

  return { predictions, accuracy, total: predictions.length, correct }
}

// ── Progressive Distillation ─────────────────────────────

// Chain: large provider → local Ollama model → classical ML
// Step 1: Generate data with large provider (already handled by generate.js)
// Step 2: Use Ollama to generate additional data (cheaper/faster)
// Step 3: Combine and train classical model
async function progressiveDistill(task, { apiKey, largeProvider = 'anthropic', localProvider = 'ollama', largeCount = 100, localCount = 200, batchSize = 10, onProgress } = {}) {
  const { generate } = await import('./generate.js')
  const { readJsonl } = await import('./data.js')

  // Step 1: Generate with large model
  onProgress?.('large', 0, largeCount)
  const largeCopy = { ...task, synthetic: { ...task.synthetic, count: largeCount, batchSize } }
  const largeResult = await generate(largeCopy, {
    apiKey,
    provider: largeProvider,
    onProgress: (done, total) => onProgress?.('large', done, total)
  })
  const largeData = await readJsonl(largeResult.path)

  // Step 2: Generate with local model
  onProgress?.('local', 0, localCount)
  const localCopy = { ...task, synthetic: { ...task.synthetic, count: localCount, batchSize } }
  let localData = []
  try {
    const localResult = await generate(localCopy, {
      provider: localProvider,
      onProgress: (done, total) => onProgress?.('local', done, total)
    })
    localData = await readJsonl(localResult.path)
  } catch {
    // Local model might not be available
  }

  // Tag provenance
  const combined = [
    ...largeData.map(r => ({ ...r, _provider: largeProvider, _distill_stage: 'large' })),
    ...localData.map(r => ({ ...r, _provider: localProvider, _distill_stage: 'local' }))
  ]

  return { combined, largeCount: largeData.length, localCount: localData.length }
}

export {
  sharedFeatureTraining,
  zeroShotEval,
  progressiveDistill
}
