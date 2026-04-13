import { join } from 'node:path'

const SCRIPTS_DIR = join(import.meta.dir, '..', 'scripts')
const MODELS_DIR = join(import.meta.dir, '..', 'models')

async function loadMeta(taskName) {
  const metaPath = join(MODELS_DIR, taskName, 'meta.json')
  const file = Bun.file(metaPath)
  if (!await file.exists()) return null
  return file.json()
}

// Predict using the Python model via subprocess
async function predict(taskName, texts) {
  const modelPath = join(MODELS_DIR, taskName, 'model.pkl')
  const scriptPath = join(SCRIPTS_DIR, 'train.py')

  const proc = Bun.spawn(['python3', scriptPath, '--predict', modelPath, '--input', '-'], {
    cwd: join(import.meta.dir, '..'),
    stdin: 'pipe',
    stdout: 'pipe',
    stderr: 'ignore'
  })

  // Write texts as JSONL to stdin
  for (const text of texts) {
    proc.stdin.write(JSON.stringify({ text }) + '\n')
  }
  proc.stdin.end()

  const stdout = await new Response(proc.stdout).text()
  const code = await proc.exited
  if (code === 0) {
    try {
      return JSON.parse(stdout)
    } catch {
      throw new Error(`Failed to parse prediction output: ${stdout}`)
    }
  }
  throw new Error(`Prediction failed (exit code ${code})`)
}

// List available models
async function listModels() {
  const { readdir } = await import('node:fs/promises')
  try {
    const dirs = await readdir(MODELS_DIR)
    const models = []
    for (const dir of dirs) {
      const meta = await loadMeta(dir)
      if (meta) models.push({ name: dir, ...meta })
    }
    return models
  } catch {
    return []
  }
}

export { predict, loadMeta, listModels }
