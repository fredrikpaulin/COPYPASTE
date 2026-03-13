// Phase 8 — Curriculum learning and data strategy
// Score example difficulty, sort by confidence, generate contrastive examples,
// cross-provider ensembling, LLM-as-judge scoring.

import { join } from 'node:path'

// ── Curriculum Learning ──────────────────────────────────

// Score difficulty by running examples through a trained model and using confidence as proxy
async function scoreDifficulty(taskName, data) {
  const { predict } = await import('./infer.js')
  const texts = data.map(d => d.text)
  const predictions = await predict(taskName, texts)

  return data.map((row, i) => ({
    ...row,
    _difficulty: predictions[i].confidence != null ? 1 - predictions[i].confidence : 0.5,
    _predicted: predictions[i].label,
    _correct: predictions[i].label === row.label
  }))
}

// Sort data by difficulty (easy first, hard last) for curriculum training
function sortByCurriculum(scoredData) {
  return [...scoredData].sort((a, b) => (a._difficulty || 0) - (b._difficulty || 0))
}

// Split into curriculum stages: easy, medium, hard
function curriculumStages(scoredData, { easyThreshold = 0.3, hardThreshold = 0.7 } = {}) {
  const easy = scoredData.filter(d => (d._difficulty || 0) < easyThreshold)
  const medium = scoredData.filter(d => (d._difficulty || 0) >= easyThreshold && (d._difficulty || 0) < hardThreshold)
  const hard = scoredData.filter(d => (d._difficulty || 0) >= hardThreshold)
  return { easy, medium, hard }
}

// ── LLM-as-Judge Quality Scoring ─────────────────────────

async function llmJudge(data, task, { apiKey, model, provider = 'anthropic', threshold = 0.7, onProgress } = {}) {
  const { callProvider } = await import('./provider.js')
  const batchSize = 15
  const batches = Math.ceil(data.length / batchSize)
  const scored = []

  for (let i = 0; i < batches; i++) {
    const batch = data.slice(i * batchSize, (i + 1) * batchSize)

    const systemPrompt = `You are a training data quality evaluator. Rate examples for a ${task.type} task: "${task.description}".`

    const userPrompt = `Rate each example on three criteria (0.0-1.0 each):
- relevance: Does the text match the task description?
- naturalness: Does the text sound like real-world data?
- label_correctness: Is the label/annotation accurate?

Examples:
${batch.map((e, idx) => `${idx}: ${JSON.stringify({ text: e.text, label: e.label || e.value })}`).join('\n')}

Respond with ONLY a JSON array: [{"index": N, "relevance": 0.0-1.0, "naturalness": 0.0-1.0, "label_correctness": 0.0-1.0}]`

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
        const scores = JSON.parse(match[0])
        for (const s of scores) {
          const avg = (s.relevance + s.naturalness + s.label_correctness) / 3
          scored.push({
            ...batch[s.index],
            _quality: avg,
            _relevance: s.relevance,
            _naturalness: s.naturalness,
            _label_correctness: s.label_correctness
          })
        }
      }
    } catch {
      // If scoring fails, keep batch with neutral score
      for (const row of batch) {
        scored.push({ ...row, _quality: 0.5 })
      }
    }

    onProgress?.(Math.min((i + 1) * batchSize, data.length), data.length)
  }

  return scored
}

function filterByQuality(scoredData, threshold = 0.7) {
  const kept = scoredData.filter(d => (d._quality || 0) >= threshold)
  return { data: kept, removed: scoredData.length - kept.length }
}

// ── Contrastive Example Generation ───────────────────────

async function generateContrastive(task, { apiKey, model, provider = 'anthropic', count = 5, onProgress } = {}) {
  const { callProvider } = await import('./provider.js')

  if (task.type !== 'classification' || !task.labels || task.labels.length < 2) {
    throw new Error('Contrastive generation requires a classification task with 2+ labels')
  }

  const examples = []
  const pairs = []
  // Generate pairs of labels
  for (let i = 0; i < task.labels.length; i++) {
    for (let j = i + 1; j < task.labels.length; j++) {
      pairs.push([task.labels[i], task.labels[j]])
    }
  }

  let done = 0
  for (const [labelA, labelB] of pairs) {
    const systemPrompt = `You are a training data generator specializing in hard, ambiguous examples for a ${task.type} task: "${task.description}".`

    const userPrompt = `Generate ${count} pairs of examples that are easily confused between "${labelA}" and "${labelB}".

For each pair:
- One example should be "${labelA}" but look like it could be "${labelB}"
- One example should be "${labelB}" but look like it could be "${labelA}"

Make them genuinely ambiguous — near the decision boundary.

Respond with ONLY a JSON array: [{"text": "...", "label": "...", "confused_with": "..."}]`

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
        const parsed = JSON.parse(match[0])
        for (const ex of parsed) {
          if (ex.text && ex.label && task.labels.includes(ex.label)) {
            examples.push({ text: ex.text, label: ex.label, _contrastive: true, _confused_with: ex.confused_with })
          }
        }
      }
    } catch {}

    done++
    onProgress?.(done, pairs.length)
  }

  return examples
}

// ── Cross-Provider Ensembling ────────────────────────────

async function ensembleGenerate(task, providers, { apiKey, count = 50, batchSize = 10, onProgress } = {}) {
  const { generate } = await import('./generate.js')
  const perProvider = Math.ceil(count / providers.length)
  const all = []

  for (let i = 0; i < providers.length; i++) {
    const provider = providers[i]
    // Temporarily override task count
    const taskCopy = { ...task, synthetic: { ...task.synthetic, count: perProvider, batchSize } }

    try {
      const result = await generate(taskCopy, {
        apiKey: provider.apiKey || apiKey,
        provider: provider.key,
        onProgress: (done, total) => {
          const base = i * perProvider
          onProgress?.(base + done, count)
        }
      })

      // Read back and tag with provider
      const { readJsonl } = await import('./data.js')
      const rows = await readJsonl(result.path)
      for (const row of rows) {
        all.push({ ...row, _provider: provider.key })
      }
    } catch {}
  }

  return all
}

export {
  scoreDifficulty, sortByCurriculum, curriculumStages,
  llmJudge, filterByQuality,
  generateContrastive,
  ensembleGenerate
}
