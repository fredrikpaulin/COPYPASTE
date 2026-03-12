import { test, expect, describe, afterEach } from 'bun:test'
import { loadConfig, resetConfigCache, DEFAULTS } from '../lib/config.js'
import { startLog, logEntry, flushLog } from '../lib/log.js'
import { join } from 'node:path'
import { rm, readdir } from 'node:fs/promises'

const LOGS_DIR = join(import.meta.dir, '..', 'logs')

afterEach(() => {
  resetConfigCache()
})

// ── Config ────────────────────────────────────────────────

describe('config', () => {
  test('returns defaults when no config file exists', async () => {
    const config = await loadConfig()
    expect(config.model).toBe(DEFAULTS.model)
    expect(config.batchSize).toBe(DEFAULTS.batchSize)
    expect(config.maxRetries).toBe(DEFAULTS.maxRetries)
    expect(config.splitRatio).toBe(DEFAULTS.splitRatio)
  })

  test('caches config after first load', async () => {
    const a = await loadConfig()
    const b = await loadConfig()
    expect(a).toBe(b) // same reference
  })

  test('reset clears cache', async () => {
    const a = await loadConfig()
    resetConfigCache()
    const b = await loadConfig()
    expect(a).not.toBe(b) // different reference
    expect(a).toEqual(b) // same values
  })

  test('DEFAULTS has all expected keys', () => {
    expect(DEFAULTS.model).toBeDefined()
    expect(DEFAULTS.batchSize).toBeDefined()
    expect(DEFAULTS.maxRetries).toBeDefined()
    expect(DEFAULTS.splitRatio).toBeDefined()
    expect(DEFAULTS.dataDir).toBeDefined()
    expect(DEFAULTS.modelsDir).toBeDefined()
  })
})

// ── Logging ───────────────────────────────────────────────

describe('logging', () => {
  test('writes structured JSONL log', async () => {
    await startLog('_test_log')
    logEntry('test_event', { key: 'value' })
    logEntry('another_event', { count: 42 })
    const logPath = await flushLog()

    expect(logPath).toContain('_test_log')
    expect(logPath).toEndWith('.jsonl')

    const text = await Bun.file(logPath).text()
    const lines = text.trim().split('\n').map(l => JSON.parse(l))
    expect(lines).toHaveLength(2)
    expect(lines[0].event).toBe('test_event')
    expect(lines[0].key).toBe('value')
    expect(lines[0].ts).toBeDefined()
    expect(lines[1].event).toBe('another_event')
    expect(lines[1].count).toBe(42)

    await rm(logPath)
  })

  test('flushLog returns null when no log started', async () => {
    const result = await flushLog()
    // After previous test flushed, no active log
    expect(result).toBeNull()
  })

  test('logEntry without data works', async () => {
    await startLog('_test_log2')
    logEntry('bare_event')
    const logPath = await flushLog()

    const text = await Bun.file(logPath).text()
    const entry = JSON.parse(text.trim())
    expect(entry.event).toBe('bare_event')
    expect(entry.ts).toBeDefined()

    await rm(logPath)
  })
})
