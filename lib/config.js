import { join } from 'node:path'

const CONFIG_PATH = join(import.meta.dir, '..', 'distill.config.json')

const DEFAULTS = {
  model: 'claude-sonnet-4-20250514',
  batchSize: 10,
  maxRetries: 3,
  splitRatio: 0.8,
  dataDir: 'data',
  modelsDir: 'models'
}

let cached = null

async function loadConfig() {
  if (cached) return cached

  const file = Bun.file(CONFIG_PATH)
  if (await file.exists()) {
    const user = await file.json()
    cached = { ...DEFAULTS, ...user }
  } else {
    cached = { ...DEFAULTS }
  }
  return cached
}

function resetConfigCache() {
  cached = null
}

export { loadConfig, resetConfigCache, DEFAULTS }
