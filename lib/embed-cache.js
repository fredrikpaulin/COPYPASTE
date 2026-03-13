// SQLite-backed embedding cache.
// Avoids re-embedding the same text when re-training or experimenting.
// Uses bun:sqlite — zero dependencies.

import { Database } from 'bun:sqlite'
import { join } from 'node:path'
import { mkdirSync } from 'node:fs'

const CACHE_DIR = join(import.meta.dir, '..', 'data')

function openCache(dbPath) {
  const db = new Database(dbPath)
  db.run(`CREATE TABLE IF NOT EXISTS embeddings (
    text_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (text_hash, model)
  )`)
  return db
}

// Simple FNV-1a hash — fast, deterministic, good distribution
function hashText(text) {
  let h = 2166136261
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i)
    h = (h * 16777619) >>> 0
  }
  return h.toString(36)
}

// Serialize float array to binary (Float32Array buffer)
function packEmbedding(arr) {
  return Buffer.from(new Float32Array(arr).buffer)
}

// Deserialize binary back to float array
function unpackEmbedding(buf) {
  return Array.from(new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4))
}

function createEmbedCache(cachePath) {
  const dbPath = cachePath || join(CACHE_DIR, 'embed_cache.sqlite')
  mkdirSync(join(dbPath, '..'), { recursive: true })
  const db = openCache(dbPath)

  const getStmt = db.prepare('SELECT embedding FROM embeddings WHERE text_hash = ? AND model = ?')
  const putStmt = db.prepare('INSERT OR REPLACE INTO embeddings (text_hash, model, embedding) VALUES (?, ?, ?)')
  const countStmt = db.prepare('SELECT COUNT(*) as count FROM embeddings WHERE model = ?')
  const clearStmt = db.prepare('DELETE FROM embeddings WHERE model = ?')

  return {
    // Get cached embedding or null
    get(text, model) {
      const row = getStmt.get(hashText(text), model)
      if (!row) return null
      return unpackEmbedding(row.embedding)
    },

    // Store an embedding
    put(text, model, embedding) {
      putStmt.run(hashText(text), model, packEmbedding(embedding))
    },

    // Batch lookup — returns { hits: Map<index, float[]>, misses: number[] }
    getBatch(texts, model) {
      const hits = new Map()
      const misses = []
      for (let i = 0; i < texts.length; i++) {
        const cached = this.get(texts[i], model)
        if (cached) {
          hits.set(i, cached)
        } else {
          misses.push(i)
        }
      }
      return { hits, misses }
    },

    // Batch store
    putBatch(texts, model, embeddings) {
      const tx = db.transaction(() => {
        for (let i = 0; i < texts.length; i++) {
          this.put(texts[i], model, embeddings[i])
        }
      })
      tx()
    },

    // Count cached embeddings for a model
    count(model) {
      return countStmt.get(model).count
    },

    // Clear cache for a model
    clear(model) {
      clearStmt.run(model)
    },

    // Close the database
    close() {
      db.close()
    }
  }
}

// Embed with caching — wraps the embed function from embed.js
async function cachedEmbed(providerName, texts, opts = {}) {
  const { embed } = await import('./embed.js')
  const model = opts.model || 'unknown'
  const cache = createEmbedCache(opts.cachePath)

  try {
    const { hits, misses } = cache.getBatch(texts, model)

    if (misses.length === 0) {
      // All cached
      opts.onProgress?.(texts.length, texts.length)
      return texts.map((_, i) => hits.get(i))
    }

    // Embed only the misses
    const missTexts = misses.map(i => texts[i])
    const newEmbeddings = await embed(providerName, missTexts, {
      ...opts,
      onProgress: (done, total) => {
        opts.onProgress?.(hits.size + done, texts.length)
      }
    })

    // Cache the new embeddings
    cache.putBatch(missTexts, model, newEmbeddings)

    // Merge hits + new into ordered result
    const result = new Array(texts.length)
    for (const [i, emb] of hits) {
      result[i] = emb
    }
    for (let j = 0; j < misses.length; j++) {
      result[misses[j]] = newEmbeddings[j]
    }

    return result
  } finally {
    cache.close()
  }
}

export { createEmbedCache, cachedEmbed, hashText, packEmbedding, unpackEmbedding }
