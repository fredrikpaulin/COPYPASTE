import { join } from 'node:path'
import { mkdir } from 'node:fs/promises'

const LOGS_DIR = join(import.meta.dir, '..', 'logs')

let logFile = null
let logEntries = []

async function startLog(taskName) {
  await mkdir(LOGS_DIR, { recursive: true })
  const ts = new Date().toISOString().replace(/[:.]/g, '-')
  logFile = join(LOGS_DIR, `${taskName}_${ts}.jsonl`)
  logEntries = []
}

function logEntry(event, data = {}) {
  const entry = { ts: new Date().toISOString(), event, ...data }
  logEntries.push(entry)
}

async function flushLog() {
  if (!logFile || !logEntries.length) return logFile
  const lines = logEntries.map(e => JSON.stringify(e)).join('\n') + '\n'
  await Bun.write(logFile, lines)
  const path = logFile
  logFile = null
  logEntries = []
  return path
}

export { startLog, logEntry, flushLog }
