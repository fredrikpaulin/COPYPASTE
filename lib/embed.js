// Embedding abstraction — supports OpenAI and Ollama embedding APIs.
// No SDKs, just fetch.

const EMBEDDING_PROVIDERS = {
  openai: {
    name: 'OpenAI Embeddings',
    envKey: 'OPENAI_API_KEY',
    defaultModel: 'text-embedding-3-small',
    defaultUrl: 'https://api.openai.com/v1/embeddings',
    dimensions: 1536
  },
  ollama: {
    name: 'Ollama Embeddings',
    envKey: null,
    defaultModel: 'nomic-embed-text',
    defaultUrl: 'http://localhost:11434/api/embed',
    dimensions: 768
  }
}

async function embedOpenAI(texts, { apiKey, model, url }) {
  const res = await fetch(url || EMBEDDING_PROVIDERS.openai.defaultUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({ model, input: texts })
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`OpenAI embeddings ${res.status}: ${body}`)
  }
  const data = await res.json()
  // data.data is sorted by index
  return data.data
    .sort((a, b) => a.index - b.index)
    .map(d => d.embedding)
}

async function embedOllama(texts, { model, url }) {
  const endpoint = url || EMBEDDING_PROVIDERS.ollama.defaultUrl
  // Ollama /api/embed supports batch via `input` array
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, input: texts })
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Ollama embeddings ${res.status}: ${body}`)
  }
  const data = await res.json()
  return data.embeddings
}

const EMBED_CALLERS = { openai: embedOpenAI, ollama: embedOllama }

// Embed a batch of texts. Returns array of float arrays.
async function embed(providerName, texts, opts = {}) {
  const caller = EMBED_CALLERS[providerName]
  if (!caller) throw new Error(`Unknown embedding provider: ${providerName}. Available: ${Object.keys(EMBED_CALLERS).join(', ')}`)

  const model = opts.model || EMBEDDING_PROVIDERS[providerName].defaultModel
  const url = opts.url || EMBEDDING_PROVIDERS[providerName].defaultUrl
  const apiKey = opts.apiKey || (EMBEDDING_PROVIDERS[providerName].envKey ? process.env[EMBEDDING_PROVIDERS[providerName].envKey] : null)

  // Batch in chunks of 100 for large datasets
  const batchSize = opts.batchSize || 100
  const all = []

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize)
    const embeddings = await caller(batch, { apiKey, model, url })
    all.push(...embeddings)
    opts.onProgress?.(Math.min(i + batchSize, texts.length), texts.length)
  }

  return all
}

// Cosine similarity between two vectors
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB)
  return denom === 0 ? 0 : dot / denom
}

// Resolve embedding provider config from task + environment
function resolveEmbedProvider(task, config = {}) {
  const providerName = task.embeddings?.provider || config.embedProvider || 'openai'
  const providerDef = EMBEDDING_PROVIDERS[providerName]
  if (!providerDef) throw new Error(`Unknown embedding provider: ${providerName}`)

  const apiKey = providerDef.envKey ? process.env[providerDef.envKey] : null
  const model = task.embeddings?.model || config.embedModel || providerDef.defaultModel
  const url = task.embeddings?.url || config.embedUrl || providerDef.defaultUrl

  return { key: providerName, name: providerDef.name, apiKey, model, url, dimensions: providerDef.dimensions }
}

function listEmbedProviders() {
  return Object.entries(EMBEDDING_PROVIDERS).map(([key, p]) => ({
    key,
    name: p.name,
    envKey: p.envKey,
    defaultModel: p.defaultModel,
    dimensions: p.dimensions,
    configured: p.envKey ? !!process.env[p.envKey] : true
  }))
}

export { embed, cosineSimilarity, resolveEmbedProvider, listEmbedProviders, EMBEDDING_PROVIDERS }
