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

export { loadAndMerge, split, stats, readJsonl, writeJsonl }
