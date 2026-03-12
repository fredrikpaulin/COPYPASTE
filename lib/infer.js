import { spawn } from 'node:child_process'
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
function predict(taskName, texts) {
  const modelPath = join(MODELS_DIR, taskName, 'model.pkl')
  const scriptPath = join(SCRIPTS_DIR, 'train.py')

  return new Promise((resolve, reject) => {
    const proc = spawn('python3', [
      scriptPath,
      '--predict', modelPath,
      '--input', '-'
    ], {
      cwd: join(import.meta.dir, '..'),
      env: { ...process.env }
    })

    // Write texts as JSONL to stdin
    for (const text of texts) {
      proc.stdin.write(JSON.stringify({ text }) + '\n')
    }
    proc.stdin.end()

    let stdout = ''
    let stderr = ''

    proc.stdout.on('data', chunk => { stdout += chunk.toString() })
    proc.stderr.on('data', chunk => { stderr += chunk.toString() })

    proc.on('close', code => {
      if (code === 0) {
        try {
          resolve(JSON.parse(stdout))
        } catch {
          reject(new Error(`Failed to parse prediction output: ${stdout}`))
        }
      } else {
        reject(new Error(`Prediction failed (code ${code}): ${stderr}`))
      }
    })

    proc.on('error', err => {
      reject(new Error(`Failed to spawn python3: ${err.message}`))
    })
  })
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
