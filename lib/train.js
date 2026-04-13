import { join } from 'node:path'
import { mkdir, cp, readdir } from 'node:fs/promises'

const MODELS_DIR = join(import.meta.dir, '..', 'models')

async function runTraining(task, trainPath, valPath, { onStdout, onStderr, onnx = false, algorithm, compare = false, search = false, grid, trainEmbeddings, valEmbeddings, dimReduce, nComponents } = {}) {
  const script = task.training?.script || 'scripts/train.py'
  const scriptPath = join(import.meta.dir, '..', script)
  const modelDir = join(MODELS_DIR, task.name)

  const args = [
    scriptPath,
    '--train', trainPath,
    '--val', valPath,
    '--output', modelDir,
    '--task-type', task.type
  ]

  if (onnx) args.push('--onnx')
  if (algorithm) args.push('--algorithm', algorithm)
  if (compare) args.push('--compare')
  if (search) args.push('--search')
  if (grid) args.push('--grid', JSON.stringify(grid))
  if (trainEmbeddings) args.push('--train-embeddings', trainEmbeddings)
  if (valEmbeddings) args.push('--val-embeddings', valEmbeddings)
  if (dimReduce) args.push('--dim-reduce', dimReduce)
  if (nComponents) args.push('--n-components', String(nComponents))

  // Pass fields for extraction tasks
  if (task.fields) {
    args.push('--fields', task.fields.join(','))
  }

  // Pass extra training args
  if (task.training?.args) {
    for (const [k, v] of Object.entries(task.training.args)) {
      args.push(`--${k}`, String(v))
    }
  }

  if (task.labels) {
    args.push('--labels', task.labels.join(','))
  }

  const proc = Bun.spawn(['python3', ...args], {
    cwd: join(import.meta.dir, '..'),
    stdout: 'pipe',
    stderr: 'pipe'
  })

  // Stream stdout and stderr concurrently
  const [stdout, stderr] = await Promise.all([
    (async () => {
      let out = ''
      for await (const chunk of proc.stdout) {
        const text = new TextDecoder().decode(chunk)
        out += text
        onStdout?.(text)
      }
      return out
    })(),
    (async () => {
      let err = ''
      for await (const chunk of proc.stderr) {
        const text = new TextDecoder().decode(chunk)
        err += text
        onStderr?.(text)
      }
      return err
    })()
  ])

  const code = await proc.exited
  if (code === 0) {
    return { modelDir, stdout, stderr }
  }
  throw new Error(`Training exited with code ${code}\n${stderr}`)
}

// Snapshot current model as a versioned copy
async function versionModel(taskName) {
  const modelDir = join(MODELS_DIR, taskName)
  const metaPath = join(modelDir, 'meta.json')
  const metaFile = Bun.file(metaPath)

  if (!await metaFile.exists()) {
    throw new Error(`No model found for task "${taskName}"`)
  }

  const meta = await metaFile.json()
  const ts = new Date().toISOString().replace(/[:.]/g, '-')
  const versionDir = join(MODELS_DIR, taskName, 'versions', ts)
  await mkdir(versionDir, { recursive: true })

  // Copy model artifacts to version dir
  const files = await readdir(modelDir)
  for (const file of files) {
    if (file === 'versions') continue
    const src = join(modelDir, file)
    const dest = join(versionDir, file)
    await cp(src, dest)
  }

  return { version: ts, path: versionDir, accuracy: meta.accuracy }
}

// List all versions for a task
async function listVersions(taskName) {
  const versionsDir = join(MODELS_DIR, taskName, 'versions')
  try {
    const dirs = await readdir(versionsDir)
    const versions = []
    for (const dir of dirs) {
      const metaFile = Bun.file(join(versionsDir, dir, 'meta.json'))
      if (await metaFile.exists()) {
        const meta = await metaFile.json()
        versions.push({ version: dir, ...meta })
      }
    }
    return versions.sort((a, b) => b.version.localeCompare(a.version))
  } catch {
    return []
  }
}

// Rollback to a specific version
async function rollbackModel(taskName, version) {
  const versionDir = join(MODELS_DIR, taskName, 'versions', version)
  const modelDir = join(MODELS_DIR, taskName)

  const files = await readdir(versionDir)
  for (const file of files) {
    await cp(join(versionDir, file), join(modelDir, file))
  }

  return { version, path: modelDir }
}

export { runTraining, versionModel, listVersions, rollbackModel }
