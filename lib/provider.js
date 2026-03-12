// Multi-provider abstraction for LLM API calls.
// Supports: anthropic (Claude), openai, ollama

const PROVIDERS = {
  anthropic: {
    name: 'Anthropic (Claude)',
    envKey: 'ANTHROPIC_API_KEY',
    defaultModel: 'claude-sonnet-4-20250514',
    defaultUrl: 'https://api.anthropic.com/v1/messages'
  },
  openai: {
    name: 'OpenAI',
    envKey: 'OPENAI_API_KEY',
    defaultModel: 'gpt-4o-mini',
    defaultUrl: 'https://api.openai.com/v1/chat/completions'
  },
  ollama: {
    name: 'Ollama (local)',
    envKey: null,
    defaultModel: 'llama3',
    defaultUrl: 'http://localhost:11434/api/chat'
  }
}

function backoffMs(attempt, base = 1000, max = 60000) {
  const exp = Math.min(base * 2 ** attempt, max)
  return exp / 2 + Math.random() * (exp / 2)
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)) }

async function callAnthropic(apiKey, model, systemPrompt, userPrompt, url) {
  const res = await fetch(url || PROVIDERS.anthropic.defaultUrl, {
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
  return { res, extractText: async () => {
    const data = await res.json()
    return data.content[0].text
  }}
}

async function callOpenAI(apiKey, model, systemPrompt, userPrompt, url) {
  const res = await fetch(url || PROVIDERS.openai.defaultUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      max_tokens: 4096,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ]
    })
  })
  return { res, extractText: async () => {
    const data = await res.json()
    return data.choices[0].message.content
  }}
}

async function callOllama(apiKey, model, systemPrompt, userPrompt, url) {
  const res = await fetch(url || PROVIDERS.ollama.defaultUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      stream: false,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ]
    })
  })
  return { res, extractText: async () => {
    const data = await res.json()
    return data.message.content
  }}
}

const CALLERS = { anthropic: callAnthropic, openai: callOpenAI, ollama: callOllama }

// Unified call with retry logic
async function callProvider(providerName, { apiKey, model, systemPrompt, userPrompt, url, maxRetries = 3, onRetry } = {}) {
  const caller = CALLERS[providerName]
  if (!caller) throw new Error(`Unknown provider: ${providerName}. Available: ${Object.keys(CALLERS).join(', ')}`)

  const actualModel = model || PROVIDERS[providerName].defaultModel

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const { res, extractText } = await caller(apiKey, actualModel, systemPrompt, userPrompt, url)

    if (res.ok) {
      return extractText()
    }

    const status = res.status
    const body = await res.text()

    // Don't retry client errors (except 429 rate limit and 529 overloaded)
    if (status !== 429 && status !== 529 && status < 500) {
      throw new Error(`${providerName} API ${status}: ${body}`)
    }

    if (attempt === maxRetries) {
      throw new Error(`${providerName} API ${status} after ${maxRetries + 1} attempts: ${body}`)
    }

    const retryAfter = res.headers.get('retry-after')
    const waitMs = retryAfter ? parseInt(retryAfter) * 1000 : backoffMs(attempt)
    onRetry?.({ attempt: attempt + 1, maxRetries, waitMs, status })
    await sleep(waitMs)
  }
}

// Resolve provider config from task + environment
function resolveProvider(task, config = {}) {
  const providerName = task.synthetic?.provider || config.provider || 'anthropic'
  const providerDef = PROVIDERS[providerName]
  if (!providerDef) throw new Error(`Unknown provider: ${providerName}`)

  const apiKey = providerDef.envKey ? process.env[providerDef.envKey] : null
  const model = task.synthetic?.model || config.model || providerDef.defaultModel
  const url = task.synthetic?.url || config.providerUrl || providerDef.defaultUrl

  return { key: providerName, name: providerDef.name, apiKey, model, url, defaultModel: providerDef.defaultModel }
}

function listProviders() {
  return Object.entries(PROVIDERS).map(([key, p]) => ({
    key,
    name: p.name,
    envKey: p.envKey,
    defaultModel: p.defaultModel,
    configured: p.envKey ? !!process.env[p.envKey] : true
  }))
}

export { callProvider, resolveProvider, listProviders, PROVIDERS, backoffMs }
