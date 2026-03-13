// Few-shot prompt optimization.
// Selects the best subset of training examples to use as few-shot demonstrations
// in LLM prompts. Supports multiple selection strategies:
//
//   random     — baseline: pick K examples at random
//   balanced   — equal representation per label/score-bucket
//   diverse    — maximize feature diversity via greedy coverage
//   similar    — pick examples most similar to the query text
//   optimized  — evaluate candidate sets against a validation set and pick the best
//
// Works across all task types: classification, sequence-labeling, scoring.

import { join } from 'node:path'

const DATA_DIR = join(import.meta.dir, '..', 'data')
const MODELS_DIR = join(import.meta.dir, '..', 'models')

// ── Text similarity (Jaccard over word n-grams) ────────

function tokenize(text) {
  return (text || '').toLowerCase().split(/\s+/).filter(Boolean)
}

function ngrams(tokens, n) {
  const grams = new Set()
  for (let i = 0; i <= tokens.length - n; i++) {
    grams.add(tokens.slice(i, i + n).join(' '))
  }
  return grams
}

function jaccard(setA, setB) {
  if (setA.size === 0 && setB.size === 0) return 1
  let inter = 0
  for (const x of setA) if (setB.has(x)) inter++
  return inter / (setA.size + setB.size - inter)
}

function textSimilarity(a, b) {
  const tokA = tokenize(a)
  const tokB = tokenize(b)
  const uni = jaccard(new Set(tokA), new Set(tokB))
  const biA = ngrams(tokA, 2)
  const biB = ngrams(tokB, 2)
  const bi = biA.size > 0 || biB.size > 0 ? jaccard(biA, biB) : uni
  return 0.6 * uni + 0.4 * bi
}

// ── Feature extraction for diversity ───────────────────

function exampleFeatures(example, task) {
  const feats = new Set()
  const text = example.text || ''
  const words = tokenize(text)

  // Word features
  for (const w of words) feats.add(`w=${w}`)

  // Label / value bucket
  if (task.type === 'classification' && example.label) {
    feats.add(`label=${example.label}`)
  }
  if ((task.type === 'scoring' || task.type === 'regression') && example.value != null) {
    feats.add(`bucket=${Math.round(example.value)}`)
  }
  if (task.type === 'sequence-labeling' && example.tags) {
    const tagSet = new Set(example.tags)
    for (const t of tagSet) feats.add(`tag=${t}`)
  }

  // Length bucket
  feats.add(`len=${Math.floor(words.length / 5) * 5}`)

  return feats
}

// ── Selection strategies ───────────────────────────────

// Random: pick K examples uniformly at random
function selectRandom(examples, { k = 5 } = {}) {
  const shuffled = [...examples]
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
  }
  return shuffled.slice(0, k)
}

// Balanced: equal representation per class / score bucket
function selectBalanced(examples, task, { k = 5 } = {}) {
  const buckets = new Map()

  for (const ex of examples) {
    let key
    if (task.type === 'classification') {
      key = ex.label || 'unknown'
    } else if (task.type === 'scoring' || task.type === 'regression') {
      key = String(Math.round(ex.value ?? 0))
    } else if (task.type === 'sequence-labeling') {
      const tagTypes = new Set((ex.tags || []).filter(t => t !== 'O'))
      key = tagTypes.size > 0 ? [...tagTypes].sort().join(',') : 'O-only'
    } else {
      key = 'default'
    }

    if (!buckets.has(key)) buckets.set(key, [])
    buckets.get(key).push(ex)
  }

  // Shuffle within each bucket
  for (const [, arr] of buckets) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[arr[i], arr[j]] = [arr[j], arr[i]]
    }
  }

  // Round-robin across buckets
  const result = []
  const keys = [...buckets.keys()]
  let idx = 0
  const pointers = new Map(keys.map(k => [k, 0]))

  while (result.length < k && idx < k * keys.length) {
    const key = keys[idx % keys.length]
    const ptr = pointers.get(key)
    if (ptr < buckets.get(key).length) {
      result.push(buckets.get(key)[ptr])
      pointers.set(key, ptr + 1)
    }
    idx++
  }

  return result.slice(0, k)
}

// Diverse: greedy set-cover over features
function selectDiverse(examples, task, { k = 5 } = {}) {
  if (examples.length <= k) return [...examples]

  const featureSets = examples.map(ex => exampleFeatures(ex, task))
  const covered = new Set()
  const selected = []
  const used = new Set()

  for (let round = 0; round < k; round++) {
    let bestIdx = -1
    let bestGain = -1

    for (let i = 0; i < examples.length; i++) {
      if (used.has(i)) continue
      let gain = 0
      for (const f of featureSets[i]) {
        if (!covered.has(f)) gain++
      }
      if (gain > bestGain) {
        bestGain = gain
        bestIdx = i
      }
    }

    if (bestIdx === -1) break
    selected.push(examples[bestIdx])
    used.add(bestIdx)
    for (const f of featureSets[bestIdx]) covered.add(f)
  }

  return selected
}

// Similar: pick examples most similar to a query text
function selectSimilar(examples, query, { k = 5 } = {}) {
  const scored = examples.map(ex => ({
    example: ex,
    similarity: textSimilarity(ex.text || '', query)
  }))
  scored.sort((a, b) => b.similarity - a.similarity)
  return scored.slice(0, k).map(s => s.example)
}

// ── Prompt formatting ──────────────────────────────────

function formatExample(example, task) {
  if (task.type === 'classification') {
    return `Input: ${example.text}\nLabel: ${example.label}`
  }
  if (task.type === 'scoring' || task.type === 'regression') {
    const val = example.value != null ? example.value.toFixed(2) : '?'
    return `Input: ${example.text}\nScore: ${val}`
  }
  if (task.type === 'sequence-labeling') {
    const tokens = example.tokens || (example.text || '').split(/\s+/)
    const tags = example.tags || []
    return `Tokens: ${JSON.stringify(tokens)}\nTags: ${JSON.stringify(tags)}`
  }
  if (task.type === 'extraction') {
    return `Input: ${example.text}\nOutput: ${JSON.stringify(example.fields || example.output || {})}`
  }
  return `Input: ${example.text}\nOutput: ${example.label || example.value || JSON.stringify(example)}`
}

function buildFewShotPrompt(task, examples, query) {
  const header = taskHeader(task)
  const demos = examples.map(ex => formatExample(ex, task)).join('\n\n')
  const queryPart = task.type === 'sequence-labeling'
    ? `Tokens: ${JSON.stringify(typeof query === 'string' ? query.split(/\s+/) : query)}`
    : `Input: ${query}`

  return `${header}\n\nExamples:\n\n${demos}\n\nNow classify:\n${queryPart}\n`
}

function taskHeader(task) {
  if (task.type === 'classification') {
    return `Task: ${task.description}\nLabels: ${(task.labels || []).join(', ')}`
  }
  if (task.type === 'scoring' || task.type === 'regression') {
    const range = task.scoreRange || { min: 0, max: 5 }
    return `Task: ${task.description}\nScore range: ${range.min} to ${range.max}`
  }
  if (task.type === 'sequence-labeling') {
    return `Task: ${task.description}\nEntity types: ${(task.labels || []).join(', ')}\nTag scheme: BIO`
  }
  return `Task: ${task.description}`
}

// ── Optimization: evaluate candidate prompt sets ───────

// Score a few-shot set by running LLM inference on validation examples
// and measuring accuracy/correlation.
async function evaluatePromptSet(task, fewShotExamples, valExamples, { apiKey, provider } = {}) {
  const { callProvider } = await import('./provider.js')
  const providerName = provider || task.synthetic?.provider || 'anthropic'
  let correct = 0
  let total = 0
  const errors = []

  for (const val of valExamples) {
    const query = val.text || (val.tokens || []).join(' ')
    const prompt = buildFewShotPrompt(task, fewShotExamples, query)

    try {
      const response = await callProvider(providerName, {
        apiKey,
        model: task.synthetic?.model,
        systemPrompt: `Respond with ONLY the answer. No explanation.`,
        userPrompt: prompt,
        maxRetries: 1
      })

      const answer = response.trim()

      if (task.type === 'classification') {
        if (answer.toLowerCase() === (val.label || '').toLowerCase()) correct++
      } else if (task.type === 'scoring' || task.type === 'regression') {
        const predicted = parseFloat(answer)
        if (!isNaN(predicted) && Math.abs(predicted - val.value) < 0.5) correct++
      } else if (task.type === 'sequence-labeling') {
        // Rough match: check if any entity types were correctly identified
        try {
          const tags = JSON.parse(answer)
          const goldEntities = new Set((val.tags || []).filter(t => t !== 'O'))
          const predEntities = new Set((Array.isArray(tags) ? tags : []).filter(t => t !== 'O'))
          const overlap = [...goldEntities].filter(t => predEntities.has(t)).length
          if (goldEntities.size > 0 && overlap / goldEntities.size >= 0.5) correct++
          else if (goldEntities.size === 0 && predEntities.size === 0) correct++
        } catch {
          errors.push({ query, response: answer, error: 'parse_error' })
        }
      }
      total++
    } catch (err) {
      errors.push({ query, error: err.message })
      total++
    }
  }

  return {
    accuracy: total > 0 ? correct / total : 0,
    correct,
    total,
    errors,
    k: fewShotExamples.length
  }
}

// Run multiple strategies and pick the best one
async function optimizeFewShot(task, trainExamples, valExamples, {
  k = 5,
  strategies = ['random', 'balanced', 'diverse'],
  randomTrials = 3,
  apiKey,
  provider,
  onStrategy
} = {}) {
  const candidates = []

  for (const strategy of strategies) {
    if (strategy === 'random') {
      // Run multiple random trials
      for (let t = 0; t < randomTrials; t++) {
        const selected = selectRandom(trainExamples, { k })
        candidates.push({ strategy: `random_${t + 1}`, examples: selected })
      }
    } else if (strategy === 'balanced') {
      candidates.push({ strategy: 'balanced', examples: selectBalanced(trainExamples, task, { k }) })
    } else if (strategy === 'diverse') {
      candidates.push({ strategy: 'diverse', examples: selectDiverse(trainExamples, task, { k }) })
    }
  }

  const results = []
  for (const candidate of candidates) {
    onStrategy?.(candidate.strategy)
    const score = await evaluatePromptSet(task, candidate.examples, valExamples, { apiKey, provider })
    results.push({ ...candidate, ...score })
  }

  results.sort((a, b) => b.accuracy - a.accuracy)
  return results
}

// ── Persistence ────────────────────────────────────────

async function saveFewShotConfig(taskName, config) {
  const dir = join(MODELS_DIR, taskName)
  const { mkdir } = await import('node:fs/promises')
  await mkdir(dir, { recursive: true })
  const path = join(dir, 'few_shot_config.json')
  await Bun.write(path, JSON.stringify(config, null, 2) + '\n')
  return path
}

async function loadFewShotConfig(taskName) {
  const path = join(MODELS_DIR, taskName, 'few_shot_config.json')
  const file = Bun.file(path)
  if (await file.exists()) return file.json()
  return null
}

// ── Quick-select: choose strategy and return examples ──

function selectExamples(strategy, examples, task, { k = 5, query } = {}) {
  switch (strategy) {
    case 'random': return selectRandom(examples, { k })
    case 'balanced': return selectBalanced(examples, task, { k })
    case 'diverse': return selectDiverse(examples, task, { k })
    case 'similar': {
      if (!query) throw new Error('similar strategy requires a query')
      return selectSimilar(examples, query, { k })
    }
    default: throw new Error(`Unknown strategy: ${strategy}`)
  }
}

export {
  // Text similarity
  tokenize, ngrams, jaccard, textSimilarity,
  // Feature extraction
  exampleFeatures,
  // Selection strategies
  selectRandom, selectBalanced, selectDiverse, selectSimilar,
  selectExamples,
  // Prompt formatting
  formatExample, buildFewShotPrompt, taskHeader,
  // Evaluation & optimization
  evaluatePromptSet, optimizeFewShot,
  // Persistence
  saveFewShotConfig, loadFewShotConfig
}
