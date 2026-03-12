import { spawn } from 'node:child_process'
import { join } from 'node:path'

const SCRIPTS_DIR = join(import.meta.dir, '..', 'scripts')

function runTraining(task, trainPath, valPath, { onStdout, onStderr }) {
  const script = task.training?.script || 'scripts/train.py'
  const scriptPath = join(import.meta.dir, '..', script)
  const modelDir = join(import.meta.dir, '..', 'models', task.name)

  const args = [
    scriptPath,
    '--train', trainPath,
    '--val', valPath,
    '--output', modelDir,
    '--task-type', task.type
  ]

  // Pass extra training args
  if (task.training?.args) {
    for (const [k, v] of Object.entries(task.training.args)) {
      args.push(`--${k}`, String(v))
    }
  }

  if (task.labels) {
    args.push('--labels', task.labels.join(','))
  }

  return new Promise((resolve, reject) => {
    const proc = spawn('python3', args, {
      cwd: join(import.meta.dir, '..'),
      env: { ...process.env }
    })

    let stdout = ''
    let stderr = ''

    proc.stdout.on('data', chunk => {
      const text = chunk.toString()
      stdout += text
      onStdout?.(text)
    })

    proc.stderr.on('data', chunk => {
      const text = chunk.toString()
      stderr += text
      onStderr?.(text)
    })

    proc.on('close', code => {
      if (code === 0) {
        resolve({ modelDir, stdout, stderr })
      } else {
        reject(new Error(`Training exited with code ${code}\n${stderr}`))
      }
    })

    proc.on('error', err => {
      reject(new Error(`Failed to spawn python3: ${err.message}`))
    })
  })
}

export { runTraining }
