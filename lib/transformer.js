// Phase 12 — Transformer distillation orchestration
// Manages fine-tuning lifecycle: dependency checks, device detection,
// model selection, training, prediction, and ONNX export.
// The actual training runs in scripts/train_transformer.py via subprocess.

import { spawn } from 'node:child_process'
import { join } from 'node:path'

const SCRIPT_PATH = join(import.meta.dir, '..', 'scripts', 'train_transformer.py')
const MODELS_DIR = join(import.meta.dir, '..', 'models')

const MODEL_PRESETS = {
  distilbert: { name: 'distilbert-base-uncased', params: '66M', description: 'DistilBERT — fast, CPU-friendly' },
  tinybert: { name: 'huawei-noah/TinyBERT_General_4L_312D', params: '14M', description: 'TinyBERT — smallest, fastest' },
  'bert-base': { name: 'bert-base-uncased', params: '110M', description: 'BERT base — larger, more capable' },
  roberta: { name: 'roberta-base', params: '125M', description: 'RoBERTa — strong general performance' },
  minilm: { name: 'microsoft/MiniLM-L12-H384-uncased', params: '33M', description: 'MiniLM — good accuracy/speed tradeoff' }
}

// Run a utility command on the transformer script and parse JSON output
function runUtility(flag) {
  return new Promise((resolve, reject) => {
    const proc = spawn('python3', [SCRIPT_PATH, flag], {
      cwd: join(import.meta.dir, '..'),
      env: { ...process.env }
    })
    let stdout = ''
    let stderr = ''
    proc.stdout.on('data', chunk => { stdout += chunk.toString() })
    proc.stderr.on('data', chunk => { stderr += chunk.toString() })
    proc.on('close', code => {
      if (code === 0) {
        try { resolve(JSON.parse(stdout.trim())) }
        catch { reject(new Error(`Failed to parse output: ${stdout}`)) }
      } else reject(new Error(`Utility failed (code ${code}): ${stderr}`))
    })
    proc.on('error', err => reject(new Error(`Failed to spawn python3: ${err.message}`)))
  })
}

// Check if torch and transformers are installed
async function checkDeps() {
  try {
    return await runUtility('--check-deps')
  } catch {
    return { torch: false, transformers: false }
  }
}

// Detect available compute device (cuda/mps/cpu)
async function detectDevice() {
  try {
    return await runUtility('--detect-device')
  } catch {
    return { device: 'unknown', info: 'detection failed' }
  }
}

// List available model presets
function listModelPresets() {
  return Object.entries(MODEL_PRESETS).map(([key, p]) => ({
    key, ...p
  }))
}

// Fine-tune a transformer model
// Returns { modelDir, accuracy, model, device, duration, stdout, stderr }
function trainTransformer(task, trainPath, valPath, {
  model = 'distilbert',
  epochs,
  lr,
  batchSize = 16,
  maxLength,
  warmupRatio = 0.1,
  weightDecay = 0.01,
  onnx = false,
  onStdout,
  onStderr,
  onEpoch,
  onTrainLog,
  onEvalLog
} = {}) {
  const modelDir = join(MODELS_DIR, task.name)

  const args = [
    SCRIPT_PATH,
    '--train', trainPath,
    '--val', valPath,
    '--output', modelDir,
    '--task-type', task.type || 'classification',
    '--model', model,
    '--batch-size', String(batchSize),
    '--warmup-ratio', String(warmupRatio),
    '--weight-decay', String(weightDecay)
  ]

  if (epochs) args.push('--epochs', String(epochs))
  if (lr) args.push('--lr', String(lr))
  if (maxLength) args.push('--max-length', String(maxLength))
  if (onnx) args.push('--onnx')

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
    let lastResult = null

    proc.stdout.on('data', chunk => {
      const text = chunk.toString()
      stdout += text

      // Parse structured output lines for TUI callbacks
      for (const line of text.split('\n').filter(Boolean)) {
        if (line.startsWith('__EPOCH_START__:')) {
          try {
            const data = JSON.parse(line.slice('__EPOCH_START__:'.length))
            onEpoch?.(data.epoch, data.total)
          } catch {}
        } else if (line.startsWith('__TRAIN_LOG__:')) {
          try {
            const data = JSON.parse(line.slice('__TRAIN_LOG__:'.length))
            onTrainLog?.(data)
          } catch {}
        } else if (line.startsWith('__EVAL_LOG__:')) {
          try {
            const data = JSON.parse(line.slice('__EVAL_LOG__:'.length))
            onEvalLog?.(data)
          } catch {}
        } else if (line.startsWith('__TRANSFORMER_RESULTS__:')) {
          try {
            lastResult = JSON.parse(line.slice('__TRANSFORMER_RESULTS__:'.length))
          } catch {}
        }

        onStdout?.(line)
      }
    })

    proc.stderr.on('data', chunk => {
      const text = chunk.toString()
      stderr += text
      onStderr?.(text)
    })

    proc.on('close', code => {
      if (code === 0) {
        resolve({
          modelDir,
          accuracy: lastResult?.accuracy,
          model: lastResult?.model,
          device: lastResult?.device,
          duration: lastResult?.duration_s,
          stdout,
          stderr
        })
      } else {
        reject(new Error(`Transformer training exited with code ${code}\n${stderr}`))
      }
    })

    proc.on('error', err => {
      reject(new Error(`Failed to spawn python3: ${err.message}`))
    })
  })
}

// Predict using a fine-tuned transformer
function predictTransformer(taskName, texts, { maxLength = 128 } = {}) {
  const modelDir = join(MODELS_DIR, taskName, 'transformer')

  return new Promise((resolve, reject) => {
    const proc = spawn('python3', [
      SCRIPT_PATH,
      '--predict', modelDir,
      '--input', '-',
      '--max-length', String(maxLength)
    ], {
      cwd: join(import.meta.dir, '..'),
      env: { ...process.env }
    })

    for (const text of texts) proc.stdin.write(JSON.stringify({ text }) + '\n')
    proc.stdin.end()

    let stdout = ''
    let stderr = ''
    proc.stdout.on('data', chunk => { stdout += chunk.toString() })
    proc.stderr.on('data', chunk => { stderr += chunk.toString() })
    proc.on('close', code => {
      if (code === 0) {
        try { resolve(JSON.parse(stdout)) }
        catch { reject(new Error(`Parse error: ${stdout}`)) }
      } else reject(new Error(`Prediction failed (code ${code}): ${stderr}`))
    })
    proc.on('error', err => reject(new Error(`Failed to spawn python3: ${err.message}`)))
  })
}

// Check if a transformer model exists for a task
async function hasTransformerModel(taskName) {
  const metaPath = join(MODELS_DIR, taskName, 'meta.json')
  const file = Bun.file(metaPath)
  if (!await file.exists()) return false
  const meta = await file.json()
  return meta.feature_mode === 'transformer'
}

export {
  checkDeps,
  detectDevice,
  listModelPresets,
  trainTransformer,
  predictTransformer,
  hasTransformerModel,
  MODEL_PRESETS
}
