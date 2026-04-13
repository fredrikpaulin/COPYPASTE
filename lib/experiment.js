// Phase 11b — Experiment tracking
// Log every training run with data fingerprint, hyperparams, accuracy, and timestamps.
// Compare experiments side by side.

import { join } from 'node:path'
import { Database } from 'bun:sqlite'

const DB_PATH = join(import.meta.dir, '..', 'data', 'experiments.sqlite')

let _db = null

function openDb() {
  if (_db) return _db
  _db = new Database(DB_PATH, { create: true })
  _db.run(`CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    algorithm TEXT,
    accuracy REAL,
    train_size INTEGER,
    val_size INTEGER,
    data_hash TEXT,
    feature_mode TEXT,
    dim_reduce TEXT,
    n_components INTEGER,
    hyperparams TEXT,
    labels TEXT,
    duration_ms INTEGER,
    notes TEXT
  )`)
  return _db
}

function closeDb() {
  if (_db) { _db.close(); _db = null }
}

// Hash a dataset for fingerprinting — fast FNV-1a over all texts
function hashDataset(rows) {
  let hash = 0x811c9dc5 | 0
  for (const row of rows) {
    const text = row.text || ''
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i)
      hash = (hash * 0x01000193) | 0
    }
    const label = row.label || row.value || ''
    for (let i = 0; i < label.length; i++) {
      hash ^= label.charCodeAt(i)
      hash = (hash * 0x01000193) | 0
    }
  }
  return (hash >>> 0).toString(16).padStart(8, '0')
}

// Record a completed experiment
function recordExperiment(entry) {
  const db = openDb()
  const stmt = db.prepare(`INSERT INTO experiments
    (task, timestamp, algorithm, accuracy, train_size, val_size, data_hash, feature_mode, dim_reduce, n_components, hyperparams, labels, duration_ms, notes)
    VALUES ($task, $timestamp, $algorithm, $accuracy, $train_size, $val_size, $data_hash, $feature_mode, $dim_reduce, $n_components, $hyperparams, $labels, $duration_ms, $notes)`)

  stmt.run({
    $task: entry.task,
    $timestamp: entry.timestamp || new Date().toISOString(),
    $algorithm: entry.algorithm || null,
    $accuracy: entry.accuracy ?? null,
    $train_size: entry.trainSize ?? null,
    $val_size: entry.valSize ?? null,
    $data_hash: entry.dataHash || null,
    $feature_mode: entry.featureMode || 'tfidf',
    $dim_reduce: entry.dimReduce || null,
    $n_components: entry.nComponents ?? null,
    $hyperparams: entry.hyperparams ? JSON.stringify(entry.hyperparams) : null,
    $labels: entry.labels ? entry.labels.join(',') : null,
    $duration_ms: entry.durationMs ?? null,
    $notes: entry.notes || null
  })

  const id = db.query('SELECT last_insert_rowid() as id').get().id
  return id
}

// List experiments for a task (newest first), optional limit
function listExperiments(taskName, { limit = 50 } = {}) {
  const db = openDb()
  const rows = db.query(`SELECT * FROM experiments WHERE task = ? ORDER BY timestamp DESC LIMIT ?`).all(taskName, limit)
  return rows.map(r => ({
    ...r,
    hyperparams: r.hyperparams ? JSON.parse(r.hyperparams) : null,
    labels: r.labels ? r.labels.split(',') : null
  }))
}

// Get a single experiment by ID
function getExperiment(id) {
  const db = openDb()
  const row = db.query('SELECT * FROM experiments WHERE id = ?').get(id)
  if (!row) return null
  return {
    ...row,
    hyperparams: row.hyperparams ? JSON.parse(row.hyperparams) : null,
    labels: row.labels ? row.labels.split(',') : null
  }
}

// Compare two experiments by ID — returns { a, b, diff }
function compareExperiments(idA, idB) {
  const a = getExperiment(idA)
  const b = getExperiment(idB)
  if (!a || !b) throw new Error('Experiment not found')

  const diff = {
    accuracyDelta: (b.accuracy || 0) - (a.accuracy || 0),
    sameData: a.data_hash === b.data_hash,
    sameAlgorithm: a.algorithm === b.algorithm,
    trainSizeDelta: (b.train_size || 0) - (a.train_size || 0),
    changes: []
  }

  if (a.algorithm !== b.algorithm) diff.changes.push(`algorithm: ${a.algorithm} → ${b.algorithm}`)
  if (a.data_hash !== b.data_hash) diff.changes.push(`data changed (${a.data_hash} → ${b.data_hash})`)
  if (a.feature_mode !== b.feature_mode) diff.changes.push(`features: ${a.feature_mode} → ${b.feature_mode}`)
  if (a.train_size !== b.train_size) diff.changes.push(`train size: ${a.train_size} → ${b.train_size}`)
  if (JSON.stringify(a.hyperparams) !== JSON.stringify(b.hyperparams)) diff.changes.push('hyperparams changed')

  return { a, b, diff }
}

// Best experiment for a task
function bestExperiment(taskName) {
  const db = openDb()
  const row = db.query('SELECT * FROM experiments WHERE task = ? ORDER BY accuracy DESC LIMIT 1').get(taskName)
  if (!row) return null
  return {
    ...row,
    hyperparams: row.hyperparams ? JSON.parse(row.hyperparams) : null,
    labels: row.labels ? row.labels.split(',') : null
  }
}

// Delete all experiments for a task
function clearExperiments(taskName) {
  const db = openDb()
  const result = db.run('DELETE FROM experiments WHERE task = ?', taskName)
  return result.changes
}

// Summary stats across all experiments for a task
function experimentStats(taskName) {
  const db = openDb()
  const row = db.query(`SELECT
    COUNT(*) as total,
    MAX(accuracy) as best_accuracy,
    MIN(accuracy) as worst_accuracy,
    AVG(accuracy) as avg_accuracy,
    COUNT(DISTINCT algorithm) as algorithms_tried,
    COUNT(DISTINCT data_hash) as data_versions
    FROM experiments WHERE task = ?`).get(taskName)
  return row
}

export {
  recordExperiment, listExperiments, getExperiment,
  compareExperiments, bestExperiment, clearExperiments,
  experimentStats, hashDataset, openDb, closeDb
}
