import { join } from 'node:path'

const DATA_DIR = join(import.meta.dir, '..', 'data')

async function readJsonl(path) {
  const text = await Bun.file(path).text()
  return text.trim().split('\n').filter(Boolean).map(line => JSON.parse(line))
}

async function writeJsonl(path, rows) {
  await Bun.write(path, rows.map(r => JSON.stringify(r)).join('\n') + '\n')
}

// Shuffle array in place (Fisher-Yates)
function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[arr[i], arr[j]] = [arr[j], arr[i]]
  }
  return arr
}

// Normalize real data fields to match synthetic format
function normalizeRealData(rows, task) {
  const inputField = task.realData?.inputField || 'text'
  const labelField = task.realData?.labelField || 'label'

  return rows.map(row => ({
    text: row[inputField],
    label: row[labelField],
    _source: 'real'
  }))
}

async function loadAndMerge(task) {
  const syntheticPath = join(DATA_DIR, `${task.name}_synthetic.jsonl`)
  let all = []

  // Load synthetic data if it exists
  const synFile = Bun.file(syntheticPath)
  if (await synFile.exists()) {
    const synthetic = await readJsonl(syntheticPath)
    all.push(...synthetic.map(r => ({ ...r, _source: r._source || 'synthetic' })))
  }

  // Load real data if configured
  if (task.realData?.path) {
    const realFile = Bun.file(task.realData.path)
    if (await realFile.exists()) {
      const raw = await readJsonl(task.realData.path)
      all.push(...normalizeRealData(raw, task))
    }
  }

  return all
}

async function split(task, data) {
  const ratio = task.training?.splitRatio || 0.8
  const shuffled = shuffle([...data])
  const splitIdx = Math.floor(shuffled.length * ratio)

  const train = shuffled.slice(0, splitIdx)
  const val = shuffled.slice(splitIdx)

  const trainPath = join(DATA_DIR, `${task.name}_train.jsonl`)
  const valPath = join(DATA_DIR, `${task.name}_val.jsonl`)

  await writeJsonl(trainPath, train)
  await writeJsonl(valPath, val)

  return { train: { count: train.length, path: trainPath }, val: { count: val.length, path: valPath } }
}

function stats(data) {
  const total = data.length
  const bySrc = { synthetic: 0, real: 0 }
  const byLabel = {}

  for (const row of data) {
    bySrc[row._source || 'synthetic']++
    const lbl = row.label || row.value || 'unknown'
    byLabel[lbl] = (byLabel[lbl] || 0) + 1
  }

  return { total, bySrc, byLabel }
}

// ── Deduplication ─────────────────────────────────────────

// Trigram set for fuzzy matching
function trigrams(str) {
  const s = str.toLowerCase().trim()
  const set = new Set()
  for (let i = 0; i <= s.length - 3; i++) {
    set.add(s.slice(i, i + 3))
  }
  return set
}

// Jaccard similarity between two trigram sets
function trigramSimilarity(a, b) {
  let intersection = 0
  for (const t of a) {
    if (b.has(t)) intersection++
  }
  const union = a.size + b.size - intersection
  return union === 0 ? 1 : intersection / union
}

function deduplicate(data, { fuzzyThreshold = 0.85 } = {}) {
  const seen = new Set()
  const kept = []
  const trigramCache = []

  for (const row of data) {
    const text = (row.text || '').trim()

    // Exact duplicate check
    if (seen.has(text)) continue
    seen.add(text)

    // Fuzzy duplicate check
    if (fuzzyThreshold < 1) {
      const rowTrigrams = trigrams(text)
      let isDup = false
      for (const existing of trigramCache) {
        if (trigramSimilarity(rowTrigrams, existing) >= fuzzyThreshold) {
          isDup = true
          break
        }
      }
      if (isDup) continue
      trigramCache.push(rowTrigrams)
    }

    kept.push(row)
  }

  return { data: kept, removed: data.length - kept.length }
}

// ── Semantic Deduplication ────────────────────────────────

// Dedup using embedding cosine similarity — catches paraphrases that trigrams miss
async function semanticDeduplicate(data, embeddings, { threshold = 0.92 } = {}) {
  const { cosineSimilarity } = await import('./embed.js')
  const kept = []
  const keptEmbeddings = []

  for (let i = 0; i < data.length; i++) {
    let isDup = false
    for (let j = 0; j < keptEmbeddings.length; j++) {
      if (cosineSimilarity(embeddings[i], keptEmbeddings[j]) >= threshold) {
        isDup = true
        break
      }
    }
    if (!isDup) {
      kept.push(data[i])
      keptEmbeddings.push(embeddings[i])
    }
  }

  return { data: kept, removed: data.length - kept.length }
}

// ── Data Augmentation ─────────────────────────────────────

const SYNONYM_MAP = {
  good: ['great', 'excellent', 'fine', 'nice', 'solid'],
  bad: ['terrible', 'awful', 'poor', 'dreadful', 'lousy'],
  big: ['large', 'huge', 'enormous', 'massive', 'vast'],
  small: ['tiny', 'little', 'minor', 'slight', 'compact'],
  fast: ['quick', 'rapid', 'speedy', 'swift', 'prompt'],
  slow: ['sluggish', 'gradual', 'unhurried', 'leisurely', 'delayed'],
  happy: ['pleased', 'glad', 'delighted', 'cheerful', 'content'],
  sad: ['unhappy', 'upset', 'disappointed', 'miserable', 'down'],
  like: ['enjoy', 'appreciate', 'love', 'prefer', 'fancy'],
  hate: ['dislike', 'detest', 'loathe', 'despise', 'abhor'],
  buy: ['purchase', 'acquire', 'get', 'obtain', 'order'],
  use: ['utilize', 'employ', 'apply', 'operate', 'handle'],
  make: ['create', 'build', 'produce', 'construct', 'develop'],
  very: ['extremely', 'highly', 'really', 'incredibly', 'remarkably'],
  nice: ['pleasant', 'lovely', 'wonderful', 'enjoyable', 'agreeable']
}

function synonymReplace(text, replaceProbability = 0.3) {
  const words = text.split(/\s+/)
  const result = words.map(word => {
    const lower = word.toLowerCase().replace(/[^a-z]/g, '')
    const syns = SYNONYM_MAP[lower]
    if (syns && Math.random() < replaceProbability) {
      const syn = syns[Math.floor(Math.random() * syns.length)]
      // Preserve original casing of first char
      const cased = word[0] === word[0].toUpperCase()
        ? syn[0].toUpperCase() + syn.slice(1)
        : syn
      return word.replace(new RegExp(lower, 'i'), cased)
    }
    return word
  })
  return result.join(' ')
}

function randomInsert(text, insertProbability = 0.15) {
  const fillers = ['actually', 'really', 'quite', 'fairly', 'rather', 'pretty', 'somewhat', 'definitely', 'certainly', 'honestly']
  const words = text.split(/\s+/)
  const result = []
  for (const word of words) {
    if (Math.random() < insertProbability) {
      result.push(fillers[Math.floor(Math.random() * fillers.length)])
    }
    result.push(word)
  }
  return result.join(' ')
}

function augment(data, { synonymProb = 0.3, insertProb = 0.15, multiplier = 2 } = {}) {
  const augmented = []
  for (const row of data) {
    // Keep original
    augmented.push(row)
    // Generate augmented copies
    for (let i = 1; i < multiplier; i++) {
      let text = row.text
      const doSynonym = Math.random() < 0.6
      const doInsert = Math.random() < 0.4
      if (doSynonym) text = synonymReplace(text, synonymProb)
      if (doInsert) text = randomInsert(text, insertProb)
      // If neither strategy fired, always apply synonym replace as fallback
      if (text === row.text) text = synonymReplace(text, synonymProb)
      if (text !== row.text) {
        augmented.push({ ...row, text, _augmented: true })
      }
    }
  }
  return augmented
}

// ── Label Balance ─────────────────────────────────────────

function labelCounts(data) {
  const counts = {}
  for (const row of data) {
    const lbl = row.label || 'unknown'
    counts[lbl] = (counts[lbl] || 0) + 1
  }
  return counts
}

function labelImbalance(data, labels) {
  const counts = labelCounts(data)
  const total = data.length
  const expected = total / labels.length
  const deficit = {}

  for (const label of labels) {
    const count = counts[label] || 0
    if (count < expected * 0.8) { // 20% tolerance
      deficit[label] = Math.ceil(expected - count)
    }
  }

  return deficit
}

// ── Confidence Filtering ──────────────────────────────────

async function filterByConfidence(data, task, { apiKey, model, provider = 'anthropic', threshold = 0.7, onProgress }) {
  const { callProvider } = await import('./provider.js')
  const batchSize = 20
  const batches = Math.ceil(data.length / batchSize)
  const scored = []

  for (let i = 0; i < batches; i++) {
    const batch = data.slice(i * batchSize, (i + 1) * batchSize)
    const systemPrompt = `You are a training data quality evaluator for a ${task.type} task: "${task.description}".`
    const userPrompt = `Rate the quality of each training example below.

For each example, respond with a JSON array of objects: {"index": N, "score": 0.0-1.0, "reason": "brief reason"}.
Score meaning: 1.0 = perfect realistic example, 0.5 = mediocre/ambiguous, 0.0 = wrong/nonsensical.

Examples:
${batch.map((e, idx) => `${idx}: ${JSON.stringify(e)}`).join('\n')}

Respond with ONLY the JSON array.`

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
          if (s.score >= threshold) {
            scored.push(batch[s.index])
          }
        }
      }
    } catch {
      // If scoring fails, keep the batch (conservative — don't drop data on API error)
      scored.push(...batch)
    }

    onProgress?.(Math.min((i + 1) * batchSize, data.length), data.length)
  }

  return { data: scored, removed: data.length - scored.length }
}

export {
  loadAndMerge, split, stats, readJsonl, writeJsonl,
  deduplicate, trigrams, trigramSimilarity,
  semanticDeduplicate,
  augment, synonymReplace, randomInsert,
  labelCounts, labelImbalance,
  filterByConfidence
}
