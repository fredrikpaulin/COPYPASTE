import { test, expect, describe, beforeAll, afterAll } from 'bun:test'
import { join } from 'node:path'
import { mkdirSync, rmSync } from 'node:fs'

const TMP = join(import.meta.dir, 'test-tmp-phase7')

beforeAll(() => {
  mkdirSync(TMP, { recursive: true })
})

afterAll(() => {
  rmSync(TMP, { recursive: true, force: true })
})

// ── lib/embed.js ──────────────────────────────────────────

describe('embed — cosineSimilarity', () => {
  test('identical vectors = 1', async () => {
    const { cosineSimilarity } = await import('../lib/embed.js')
    const a = [1, 0, 0]
    expect(cosineSimilarity(a, a)).toBeCloseTo(1.0, 5)
  })

  test('orthogonal vectors = 0', async () => {
    const { cosineSimilarity } = await import('../lib/embed.js')
    expect(cosineSimilarity([1, 0, 0], [0, 1, 0])).toBeCloseTo(0.0, 5)
  })

  test('opposite vectors = -1', async () => {
    const { cosineSimilarity } = await import('../lib/embed.js')
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1.0, 5)
  })

  test('similar vectors are high', async () => {
    const { cosineSimilarity } = await import('../lib/embed.js')
    expect(cosineSimilarity([1, 2, 3], [1.1, 2.1, 3.1])).toBeGreaterThan(0.99)
  })

  test('zero vector returns 0', async () => {
    const { cosineSimilarity } = await import('../lib/embed.js')
    expect(cosineSimilarity([0, 0, 0], [1, 2, 3])).toBe(0)
  })
})

describe('embed — resolveEmbedProvider', () => {
  test('defaults to openai', async () => {
    const { resolveEmbedProvider } = await import('../lib/embed.js')
    const p = resolveEmbedProvider({})
    expect(p.key).toBe('openai')
    expect(p.name).toBe('OpenAI Embeddings')
  })

  test('reads from task config', async () => {
    const { resolveEmbedProvider } = await import('../lib/embed.js')
    const p = resolveEmbedProvider({ embeddings: { provider: 'ollama', model: 'mxbai-embed-large' } })
    expect(p.key).toBe('ollama')
    expect(p.model).toBe('mxbai-embed-large')
  })

  test('throws on unknown provider', async () => {
    const { resolveEmbedProvider } = await import('../lib/embed.js')
    expect(() => resolveEmbedProvider({ embeddings: { provider: 'fake' } })).toThrow('Unknown embedding provider')
  })
})

describe('embed — listEmbedProviders', () => {
  test('lists openai and ollama', async () => {
    const { listEmbedProviders } = await import('../lib/embed.js')
    const providers = listEmbedProviders()
    expect(providers.length).toBe(2)
    expect(providers.map(p => p.key)).toEqual(['openai', 'ollama'])
  })
})

describe('embed — mock provider integration', () => {
  let server

  beforeAll(() => {
    server = Bun.serve({
      port: 0,
      fetch(req) {
        return new Response(JSON.stringify({
          data: [
            { index: 0, embedding: [0.1, 0.2, 0.3] },
            { index: 1, embedding: [0.4, 0.5, 0.6] }
          ]
        }))
      }
    })
  })

  afterAll(() => {
    server.stop()
  })

  test('embed calls OpenAI API and returns embeddings', async () => {
    const { embed } = await import('../lib/embed.js')
    const result = await embed('openai', ['hello', 'world'], {
      apiKey: 'test-key',
      model: 'test-model',
      url: `http://localhost:${server.port}`,
      batchSize: 10
    })
    expect(result.length).toBe(2)
    expect(result[0]).toEqual([0.1, 0.2, 0.3])
    expect(result[1]).toEqual([0.4, 0.5, 0.6])
  })

  test('embed calls onProgress', async () => {
    const { embed } = await import('../lib/embed.js')
    let progressCalled = false
    await embed('openai', ['hello', 'world'], {
      apiKey: 'test-key',
      model: 'test-model',
      url: `http://localhost:${server.port}`,
      onProgress: () => { progressCalled = true }
    })
    expect(progressCalled).toBe(true)
  })
})

describe('embed — mock Ollama integration', () => {
  let server

  beforeAll(() => {
    server = Bun.serve({
      port: 0,
      fetch(req) {
        return new Response(JSON.stringify({
          embeddings: [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]
        }))
      }
    })
  })

  afterAll(() => {
    server.stop()
  })

  test('embed calls Ollama API and returns embeddings', async () => {
    const { embed } = await import('../lib/embed.js')
    const result = await embed('ollama', ['hello', 'world'], {
      model: 'nomic-embed-text',
      url: `http://localhost:${server.port}`
    })
    expect(result.length).toBe(2)
    expect(result[0]).toEqual([0.7, 0.8, 0.9])
  })
})

// ── lib/embed-cache.js ────────────────────────────────────

describe('embed-cache', () => {
  test('stores and retrieves embeddings', async () => {
    const { createEmbedCache } = await import('../lib/embed-cache.js')
    const dbPath = join(TMP, 'cache-test.sqlite')
    const cache = createEmbedCache(dbPath)

    cache.put('hello world', 'test-model', [0.1, 0.2, 0.3])
    const result = cache.get('hello world', 'test-model')
    expect(result).not.toBeNull()
    expect(result[0]).toBeCloseTo(0.1, 4)
    expect(result[1]).toBeCloseTo(0.2, 4)
    expect(result[2]).toBeCloseTo(0.3, 4)

    cache.close()
  })

  test('returns null for missing entries', async () => {
    const { createEmbedCache } = await import('../lib/embed-cache.js')
    const dbPath = join(TMP, 'cache-miss.sqlite')
    const cache = createEmbedCache(dbPath)

    const result = cache.get('nonexistent', 'model')
    expect(result).toBeNull()

    cache.close()
  })

  test('separates by model', async () => {
    const { createEmbedCache } = await import('../lib/embed-cache.js')
    const dbPath = join(TMP, 'cache-models.sqlite')
    const cache = createEmbedCache(dbPath)

    cache.put('text', 'model-a', [1, 2, 3])
    cache.put('text', 'model-b', [4, 5, 6])

    const a = cache.get('text', 'model-a')
    const b = cache.get('text', 'model-b')
    expect(a[0]).toBeCloseTo(1, 4)
    expect(b[0]).toBeCloseTo(4, 4)

    cache.close()
  })

  test('getBatch returns hits and misses', async () => {
    const { createEmbedCache } = await import('../lib/embed-cache.js')
    const dbPath = join(TMP, 'cache-batch.sqlite')
    const cache = createEmbedCache(dbPath)

    cache.put('cached-text', 'model', [0.5, 0.6])

    const { hits, misses } = cache.getBatch(['cached-text', 'new-text'], 'model')
    expect(hits.size).toBe(1)
    expect(misses).toEqual([1])
    expect(hits.get(0)[0]).toBeCloseTo(0.5, 4)

    cache.close()
  })

  test('putBatch stores multiple embeddings', async () => {
    const { createEmbedCache } = await import('../lib/embed-cache.js')
    const dbPath = join(TMP, 'cache-putbatch.sqlite')
    const cache = createEmbedCache(dbPath)

    cache.putBatch(['a', 'b'], 'model', [[1, 2], [3, 4]])
    expect(cache.count('model')).toBe(2)
    expect(cache.get('a', 'model')[0]).toBeCloseTo(1, 4)

    cache.close()
  })

  test('count returns correct count', async () => {
    const { createEmbedCache } = await import('../lib/embed-cache.js')
    const dbPath = join(TMP, 'cache-count.sqlite')
    const cache = createEmbedCache(dbPath)

    expect(cache.count('model')).toBe(0)
    cache.put('a', 'model', [1])
    cache.put('b', 'model', [2])
    expect(cache.count('model')).toBe(2)

    cache.close()
  })

  test('clear removes entries for a model', async () => {
    const { createEmbedCache } = await import('../lib/embed-cache.js')
    const dbPath = join(TMP, 'cache-clear.sqlite')
    const cache = createEmbedCache(dbPath)

    cache.put('a', 'model-x', [1])
    cache.put('b', 'model-y', [2])
    cache.clear('model-x')
    expect(cache.count('model-x')).toBe(0)
    expect(cache.count('model-y')).toBe(1)

    cache.close()
  })
})

describe('embed-cache — pack/unpack', () => {
  test('round-trips float arrays', async () => {
    const { packEmbedding, unpackEmbedding } = await import('../lib/embed-cache.js')
    const original = [0.123456, -0.789012, 1.5, 0.0, -3.14]
    const packed = packEmbedding(original)
    const unpacked = unpackEmbedding(packed)
    expect(unpacked.length).toBe(original.length)
    for (let i = 0; i < original.length; i++) {
      expect(unpacked[i]).toBeCloseTo(original[i], 4)
    }
  })
})

// ── Semantic deduplication ────────────────────────────────

describe('semanticDeduplicate', () => {
  test('removes near-duplicate embeddings', async () => {
    const { semanticDeduplicate } = await import('../lib/data.js')

    const data = [
      { text: 'great product', label: 'positive' },
      { text: 'excellent product', label: 'positive' },  // semantically similar
      { text: 'terrible product', label: 'negative' }
    ]
    // First two have very similar embeddings, third is different
    const embeddings = [
      [0.9, 0.1, 0.0],
      [0.89, 0.11, 0.01],  // very close to first
      [-0.9, 0.1, 0.0]     // very different
    ]

    const result = await semanticDeduplicate(data, embeddings, { threshold: 0.99 })
    expect(result.removed).toBe(1)
    expect(result.data.length).toBe(2)
  })

  test('keeps all with low threshold', async () => {
    const { semanticDeduplicate } = await import('../lib/data.js')

    const data = [
      { text: 'a', label: 'x' },
      { text: 'b', label: 'y' }
    ]
    const embeddings = [
      [1, 0],
      [0, 1]
    ]

    const result = await semanticDeduplicate(data, embeddings, { threshold: 0.99 })
    expect(result.removed).toBe(0)
    expect(result.data.length).toBe(2)
  })
})

// ── Embedding-based training (Python integration) ────────

describe('embedding training via Python', () => {
  const trainPath = join(TMP, 'embed_train.jsonl')
  const valPath = join(TMP, 'embed_val.jsonl')
  const trainEmbPath = join(TMP, 'embed_train_emb.jsonl')
  const valEmbPath = join(TMP, 'embed_val_emb.jsonl')
  const outputDir = join(TMP, 'embed_model')

  beforeAll(async () => {
    // Create tiny training data with pre-computed embeddings
    const trainRows = [
      { text: 'great product love it', label: 'positive' },
      { text: 'wonderful amazing', label: 'positive' },
      { text: 'love this so much', label: 'positive' },
      { text: 'best ever', label: 'positive' },
      { text: 'terrible awful bad', label: 'negative' },
      { text: 'horrible experience', label: 'negative' },
      { text: 'worst product ever', label: 'negative' },
      { text: 'hate it broken', label: 'negative' }
    ]
    const valRows = [
      { text: 'really good stuff', label: 'positive' },
      { text: 'very bad terrible', label: 'negative' }
    ]

    // Simple fake embeddings — positive texts get [high, low], negative get [low, high]
    const trainEmb = trainRows.map(r =>
      ({ embedding: r.label === 'positive' ? [0.9 + Math.random() * 0.1, 0.1 + Math.random() * 0.1] : [0.1 + Math.random() * 0.1, 0.9 + Math.random() * 0.1] })
    )
    const valEmb = valRows.map(r =>
      ({ embedding: r.label === 'positive' ? [0.95, 0.05] : [0.05, 0.95] })
    )

    await Bun.write(trainPath, trainRows.map(r => JSON.stringify(r)).join('\n') + '\n')
    await Bun.write(valPath, valRows.map(r => JSON.stringify(r)).join('\n') + '\n')
    await Bun.write(trainEmbPath, trainEmb.map(r => JSON.stringify(r)).join('\n') + '\n')
    await Bun.write(valEmbPath, valEmb.map(r => JSON.stringify(r)).join('\n') + '\n')
  })

  test('trains with pre-computed embeddings', async () => {
    const { spawn } = await import('node:child_process')
    const scriptPath = join(import.meta.dir, '..', 'scripts', 'train.py')

    const result = await new Promise((resolve, reject) => {
      let stdout = ''
      const proc = spawn('python3', [
        scriptPath,
        '--train', trainPath,
        '--val', valPath,
        '--output', outputDir,
        '--task-type', 'classification',
        '--labels', 'positive,negative',
        '--train-embeddings', trainEmbPath,
        '--val-embeddings', valEmbPath
      ])
      proc.stdout.on('data', d => { stdout += d.toString() })
      proc.stderr.on('data', d => {})
      proc.on('close', code => resolve({ code, stdout }))
      proc.on('error', reject)
    })

    expect(result.code).toBe(0)
    expect(result.stdout).toContain('Features: embeddings')
    expect(result.stdout).toContain('Validation Accuracy')

    // Check meta.json
    const meta = await Bun.file(join(outputDir, 'meta.json')).json()
    expect(meta.feature_mode).toBe('embeddings')
    expect(meta.accuracy).toBeGreaterThan(0)
  }, 30000)

  test('trains with embeddings + PCA reduction', async () => {
    const { spawn } = await import('node:child_process')
    const scriptPath = join(import.meta.dir, '..', 'scripts', 'train.py')
    const pcaOutputDir = join(TMP, 'embed_model_pca')

    const result = await new Promise((resolve, reject) => {
      let stdout = ''
      const proc = spawn('python3', [
        scriptPath,
        '--train', trainPath,
        '--val', valPath,
        '--output', pcaOutputDir,
        '--task-type', 'classification',
        '--labels', 'positive,negative',
        '--train-embeddings', trainEmbPath,
        '--val-embeddings', valEmbPath,
        '--dim-reduce', 'pca',
        '--n-components', '1'
      ])
      proc.stdout.on('data', d => { stdout += d.toString() })
      proc.stderr.on('data', d => {})
      proc.on('close', code => resolve({ code, stdout }))
      proc.on('error', reject)
    })

    expect(result.code).toBe(0)
    expect(result.stdout).toContain('Reducing dimensions')

    const meta = await Bun.file(join(pcaOutputDir, 'meta.json')).json()
    expect(meta.dim_reduce).toBe('pca')
  }, 30000)
})
