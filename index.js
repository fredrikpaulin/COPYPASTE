import {
  clear, banner, header, success, warn, error, info, dim,
  progress, spinner, prompt, menu, table, streamBox, SHOW_CURSOR
} from './lib/tui.js'
import { loadTask, listTasks, saveTask } from './lib/task.js'
import { generate, preview } from './lib/generate.js'
import { loadAndMerge, split, stats, deduplicate, augment, labelImbalance, filterByConfidence, semanticDeduplicate } from './lib/data.js'
import { runTraining, versionModel, listVersions } from './lib/train.js'
import { predict, loadMeta, listModels } from './lib/infer.js'
import { bundle } from './lib/bundle.js'
import { getUncertainExamples, generateAndRankByUncertainty, llmLabel, loadHistory, saveIteration } from './lib/active.js'
import { listProviders, resolveProvider } from './lib/provider.js'
import { resolveEmbedProvider, listEmbedProviders } from './lib/embed.js'
import { cachedEmbed, createEmbedCache } from './lib/embed-cache.js'
import { listTemplates, loadTemplate } from './lib/templates.js'
import { generateReport } from './lib/report.js'
import { scoreDifficulty, sortByCurriculum, curriculumStages, llmJudge, filterByQuality, generateContrastive, ensembleGenerate } from './lib/curriculum.js'
import { zeroShotEval, progressiveDistill } from './lib/multitask.js'
import { kFoldCV, featureImportance, errorTaxonomy, calibrationBins, projectTo2D } from './lib/evaluate.js'
import { trainEnsembleModels, listEnsembleModels, ensemblePredict, predictWithThreshold } from './lib/ensemble.js'
import { recordExperiment, listExperiments, compareExperiments, bestExperiment, experimentStats, hashDataset } from './lib/experiment.js'
import { checkDeps, detectDevice, listModelPresets, trainTransformer, predictTransformer, hasTransformerModel } from './lib/transformer.js'
import { trainCRF, predictSequence, predictBatch, evaluateEntities, extractEntities, saveModel as saveCRFModel, loadModel as loadCRFModel, hasCRFModel, labelsToBIO, validateBIO } from './lib/crf.js'
import { trainScoring, predictScore, predictScoreBatch, evaluateScoring, saveScoringModel, loadScoringModel, hasScoringModel } from './lib/scoring.js'
import { computeUncertainty, selectMostUncertain, activeLoop, saveActiveIteration, loadActiveHistory, appendLabeledData } from './lib/active-loop.js'
import { selectExamples, buildFewShotPrompt, optimizeFewShot, saveFewShotConfig, loadFewShotConfig, formatExample } from './lib/few-shot.js'
import { loadConfig } from './lib/config.js'
import { startLog, logEntry, flushLog } from './lib/log.js'
import { join } from 'node:path'

const API_KEY = process.env.ANTHROPIC_API_KEY

// Graceful exit
process.on('SIGINT', () => { process.stdout.write(SHOW_CURSOR); process.exit(0) })

async function createTask() {
  header('Create New Task')

  const name = await prompt('Task name (lowercase, hyphens):')
  const typeIdx = await menu('Task type:', ['classification', 'extraction', 'scoring', 'sequence-labeling'])
  const type = ['classification', 'extraction', 'scoring', 'sequence-labeling'][typeIdx]
  const description = await prompt('Description:')

  const task = { name, type, description }

  if (type === 'classification' || type === 'sequence-labeling') {
    const labelsStr = await prompt('Labels (comma-separated):')
    task.labels = labelsStr.split(',').map(s => s.trim())
  }
  if (type === 'extraction') {
    const fieldsStr = await prompt('Fields to extract (comma-separated):')
    task.fields = fieldsStr.split(',').map(s => s.trim())
  }
  if (type === 'scoring') {
    const minStr = await prompt('Minimum score [0]:')
    const maxStr = await prompt('Maximum score [5]:')
    task.scoreRange = { min: parseFloat(minStr) || 0, max: parseFloat(maxStr) || 5 }
  }

  const genPrompt = await prompt('Generation prompt template:')
  const countStr = await prompt('How many synthetic examples? [100]:')
  task.synthetic = {
    count: parseInt(countStr) || 100,
    prompt: genPrompt,
    batchSize: 10
  }

  const realPath = await prompt('Path to real data JSONL (or press enter to skip):')
  if (realPath) {
    task.realData = { path: realPath }
  }

  try {
    const path = await saveTask(task)
    success(`Task saved to ${path}`)
  } catch (e) {
    error(e.message)
  }
}

async function createFromTemplate() {
  header('Create from Template')

  const templates = await listTemplates()
  if (!templates.length) {
    warn('No templates found in templates/')
    return
  }

  const idx = await menu('Choose a template:', templates.map(t => `${t.name} (${t.type})`))
  const template = templates[idx]

  info(`Template: ${template.name} — ${template.description}`)
  const name = await prompt(`Task name [${template.name}]:`)
  const taskName = name || template.name

  const task = { ...template, name: taskName }
  delete task.file

  try {
    const path = await saveTask(task)
    success(`Task saved to ${path}`)
  } catch (e) {
    error(e.message)
  }
}

async function runReport(task) {
  header(`Evaluation report for "${task.name}"`)

  const meta = await loadMeta(task.name)
  if (!meta) {
    warn('No trained model found. Train one first.')
    return
  }

  // Load validation data and run predictions
  const { readJsonl } = await import('./lib/data.js')
  const { join } = await import('node:path')
  const valPath = join('data', `${task.name}_val.jsonl`)
  let valData
  try {
    valData = await readJsonl(valPath)
  } catch {
    warn('No validation data found. Run prepare + train first.')
    return
  }

  const sp = spinner('Running predictions on validation set...')
  const predictions = await predict(task.name, valData.map(d => d.text))
  sp.stop(`${predictions.length} predictions`)

  const labels = meta.labels || task.labels || [...new Set(valData.map(d => d.label))]
  const result = await generateReport(task.name, { valData, predictions, labels, meta })

  success(`Report saved to ${result.path}`)
  info(`Open in browser: file://${result.path}`)
}

async function runPreview(task, useStream = false) {
  if (!API_KEY) { error('Set ANTHROPIC_API_KEY in your environment'); return }
  if (!task.synthetic) { warn('No synthetic config on this task'); return }

  header(`Preview for "${task.name}"`)
  if (useStream) info('Streaming: enabled')

  let box = null
  const sp = useStream ? null : spinner('Generating sample...')

  try {
    const result = await preview(task, {
      apiKey: API_KEY,
      count: 5,
      stream: useStream,
      onToken: useStream ? (token) => {
        if (!box) box = streamBox('Streaming preview')
        box.write(token)
      } : undefined,
      onRetry: ({ attempt, waitMs }) => {
        if (box) { box.end(); box = null }
        if (sp) sp.stop(`Retry ${attempt} in ${Math.round(waitMs / 1000)}s...`)
        else warn(`Retry ${attempt} in ${Math.round(waitMs / 1000)}s...`)
      }
    })
    if (box) { box.end(); box = null }
    if (sp) sp.stop(`${result.examples.length} examples generated`)

    for (const ex of result.examples) {
      info(JSON.stringify(ex))
    }
    if (result.dropped) {
      warn(`${result.dropped} malformed examples dropped`)
    }
  } catch (e) {
    if (box) { box.end(); box = null }
    if (sp) sp.fail('Preview failed')
    error(e.message)
  }
}

async function runGenerate(task, useStream = false) {
  if (!API_KEY) { error('Set ANTHROPIC_API_KEY in your environment'); return }
  if (!task.synthetic) { warn('No synthetic config on this task'); return }

  await startLog(task.name)
  logEntry('generate_start', { task: task.name, count: task.synthetic.count, stream: useStream })

  header(`Generating synthetic data for "${task.name}"`)
  info(`Model: ${task.synthetic.model || 'claude-sonnet-4-20250514'}`)
  info(`Target: ${task.synthetic.count} examples in batches of ${task.synthetic.batchSize}`)
  if (useStream) info('Streaming: enabled — tokens will appear as they arrive')

  let box = null
  try {
    const result = await generate(task, {
      apiKey: API_KEY,
      stream: useStream,
      onToken: useStream ? (token, full, { batch, batches }) => {
        if (!box) box = streamBox(`Batch ${batch}/${batches}`)
        box.write(token)
      } : undefined,
      onProgress: (current, total) => {
        if (box) { box.end(); box = null }
        progress(current, total, 'examples')
      },
      onRetry: ({ attempt, waitMs, status }) => {
        if (box) { box.end(); box = null }
        warn(`Rate limited (${status}), retry ${attempt} in ${Math.round(waitMs / 1000)}s...`)
        logEntry('retry', { attempt, waitMs, status })
      },
      onDropped: count => {
        warn(`${count} malformed examples dropped during validation`)
        logEntry('dropped', { count })
      }
    })
    if (box) { box.end(); box = null }
    success(`Generated ${result.count} examples → ${result.path}`)
    if (result.dropped) dim(`  (${result.dropped} dropped)`)
    logEntry('generate_complete', { count: result.count, dropped: result.dropped })
  } catch (e) {
    if (box) { box.end(); box = null }
    error(`Generation failed: ${e.message}`)
    logEntry('generate_error', { error: e.message })
  }

  const logPath = await flushLog()
  if (logPath) dim(`  Log: ${logPath}`)
}

async function runPrepare(task) {
  header(`Preparing data for "${task.name}"`)

  let data = await loadAndMerge(task)
  if (!data.length) {
    warn('No data found. Generate synthetic data first.')
    return null
  }

  // Deduplication
  const dedupResult = deduplicate(data)
  if (dedupResult.removed > 0) {
    info(`Deduplicated: removed ${dedupResult.removed} duplicates`)
    data = dedupResult.data
  }

  const s = stats(data)
  table(
    [
      ['Total', s.total],
      ['Synthetic', s.bySrc.synthetic],
      ['Real', s.bySrc.real],
      ...Object.entries(s.byLabel).map(([k, v]) => [`  label: ${k}`, v])
    ],
    ['Metric', 'Count']
  )

  // Label imbalance check
  if (task.labels) {
    const deficit = labelImbalance(data, task.labels)
    if (Object.keys(deficit).length > 0) {
      warn('Label imbalance detected:')
      for (const [label, needed] of Object.entries(deficit)) {
        dim(`  ${label}: needs ~${needed} more examples`)
      }
    }
  }

  const result = await split(task, data)
  success(`Train: ${result.train.count} → ${result.train.path}`)
  success(`Val: ${result.val.count} → ${result.val.path}`)
  return result
}

async function runAugment(task) {
  header(`Augmenting data for "${task.name}"`)

  const data = await loadAndMerge(task)
  if (!data.length) { warn('No data found.'); return }

  const multiplierStr = await prompt('Augmentation multiplier [2]:')
  const multiplier = parseInt(multiplierStr) || 2

  const augmented = augment(data, { multiplier })
  const added = augmented.length - data.length
  success(`Augmented: ${data.length} → ${augmented.length} (+${added} examples)`)

  // Show label distribution after augmentation
  const s = stats(augmented)
  table(
    Object.entries(s.byLabel).map(([k, v]) => [k, v]),
    ['Label', 'Count']
  )
}

async function runConfidenceFilter(task) {
  if (!API_KEY) { error('Set ANTHROPIC_API_KEY in your environment'); return }

  header(`Confidence filtering for "${task.name}"`)

  const data = await loadAndMerge(task)
  if (!data.length) { warn('No data found.'); return }

  const thresholdStr = await prompt('Confidence threshold (0.0-1.0) [0.7]:')
  const threshold = parseFloat(thresholdStr) || 0.7

  info(`Filtering ${data.length} examples with threshold ${threshold}...`)

  const result = await filterByConfidence(data, task, {
    apiKey: API_KEY,
    threshold,
    onProgress: (cur, total) => progress(cur, total, 'scored')
  })

  success(`Kept ${result.data.length}/${data.length} examples (${result.removed} removed)`)
}

async function runTrain(task, splitResult, { algorithm, compare = false, search = false, grid, trainEmbeddings, valEmbeddings, dimReduce, nComponents } = {}) {
  header(`Training model for "${task.name}"`)

  // Version existing model before overwriting
  const meta = await loadMeta(task.name)
  if (meta) {
    try {
      const v = await versionModel(task.name)
      dim(`  Previous model versioned: ${v.version} (accuracy: ${v.accuracy})`)
    } catch {}
  }

  const label = compare ? 'Comparing algorithms...' : search ? 'Searching hyperparameters...' : 'Training in progress...'
  const sp = spinner(label)
  const startTime = Date.now()
  try {
    const result = await runTraining(task, splitResult.train.path, splitResult.val.path, {
      onnx: !compare && !search,
      algorithm,
      compare,
      search,
      grid,
      trainEmbeddings,
      valEmbeddings,
      dimReduce,
      nComponents,
      onStdout: text => {
        for (const line of text.split('\n').filter(Boolean)) {
          if (line.includes('Accuracy')) {
            sp.stop(line.trim())
          }
          if (line.startsWith('__COMPARE_RESULTS__:')) {
            const results = JSON.parse(line.slice('__COMPARE_RESULTS__:'.length))
            info('\nComparison:')
            table(
              results.map((r, i) => [r.algorithm, r.accuracy.toFixed(4), i === 0 ? '← best' : '']),
              ['Algorithm', 'Accuracy', '']
            )
          }
          if (line.startsWith('__SEARCH_RESULTS__:')) {
            const results = JSON.parse(line.slice('__SEARCH_RESULTS__:'.length))
            info('\nSearch results (top 5):')
            table(
              results.slice(0, 5).map((r, i) => [JSON.stringify(r.params), r.accuracy.toFixed(4), i === 0 ? '← best' : '']),
              ['Params', 'Accuracy', '']
            )
          }
        }
      },
      onStderr: text => dim(text.trim())
    })
    const durationMs = Date.now() - startTime
    success(`Model saved to ${result.modelDir}`)

    // Record experiment automatically
    if (!compare && !search) {
      try {
        const { readJsonl } = await import('./lib/data.js')
        const trainData = await readJsonl(splitResult.train.path)
        const valData = await readJsonl(splitResult.val.path)
        const newMeta = await loadMeta(task.name)
        recordExperiment({
          task: task.name,
          algorithm: algorithm || 'logistic_regression',
          accuracy: newMeta?.accuracy ?? null,
          trainSize: trainData.length,
          valSize: valData.length,
          dataHash: hashDataset(trainData),
          featureMode: trainEmbeddings ? 'embeddings' : 'tfidf',
          dimReduce: dimReduce || null,
          nComponents: nComponents ?? null,
          labels: task.labels || null,
          durationMs
        })
        dim('  Experiment recorded.')
      } catch {}
    }
  } catch (e) {
    sp.fail('Training failed')
    error(e.message)
  }
}

async function runCompare(task) {
  const splitResult = await runPrepare(task)
  if (!splitResult) return
  await runTrain(task, splitResult, { compare: true })
}

async function runHyperparamSearch(task) {
  const algoIdx = await menu('Algorithm to search:', ['logistic_regression', 'svm', 'random_forest'])
  const algorithm = ['logistic_regression', 'svm', 'random_forest'][algoIdx]

  const splitResult = await runPrepare(task)
  if (!splitResult) return
  await runTrain(task, splitResult, { algorithm, search: true })
}

async function runPredict(task) {
  header(`Predict with "${task.name}" model`)

  const meta = await loadMeta(task.name)
  if (!meta) {
    warn('No trained model found. Train one first.')
    return
  }

  info(`Model: ${meta.task_type} | Accuracy: ${meta.accuracy?.toFixed(4)} | Labels: ${meta.labels?.join(', ')}`)

  while (true) {
    const text = await prompt('Enter text (or "back" to return):')
    if (text.toLowerCase() === 'back') return

    try {
      const results = await predict(task.name, [text])
      const r = results[0]
      success(`Label: ${r.label}  (confidence: ${(r.confidence * 100).toFixed(1)}%)`)
    } catch (e) {
      error(`Prediction failed: ${e.message}`)
    }
  }
}

async function runModelVersions(task) {
  header(`Model versions for "${task.name}"`)

  const versions = await listVersions(task.name)
  if (!versions.length) {
    warn('No version history yet.')
    return
  }

  table(
    versions.map(v => [v.version, v.accuracy?.toFixed(4) || 'N/A', v.train_size || '', v.val_size || '']),
    ['Version', 'Accuracy', 'Train', 'Val']
  )
}

async function runFullPipeline(task) {
  await runGenerate(task)
  const splitResult = await runPrepare(task)
  if (splitResult) {
    await runTrain(task, splitResult)
  }
}

async function runBundle(task) {
  header(`Bundle model for "${task.name}"`)

  const meta = await loadMeta(task.name)
  if (!meta) {
    warn('No trained model found. Train one first.')
    return
  }

  const outputDir = await prompt(`Output directory [dist/${task.name}]:`)
  const dir = outputDir || `dist/${task.name}`

  try {
    const result = await bundle(task.name, dir)
    success(`Bundled to ${result.path}`)
    dim(`  Files: ${result.files.join(', ')}`)
  } catch (e) {
    error(`Bundle failed: ${e.message}`)
  }
}

async function runUncertaintySampling(task) {
  if (!API_KEY) { error('Set ANTHROPIC_API_KEY in your environment'); return }

  const meta = await loadMeta(task.name)
  if (!meta) {
    warn('No trained model found. Train one first.')
    return
  }

  header(`Uncertainty sampling for "${task.name}"`)
  info('Generating candidate examples and ranking by model uncertainty...')

  const sp = spinner('Generating candidates...')
  try {
    const result = await generateAndRankByUncertainty(task, {
      apiKey: API_KEY,
      count: 20,
      topK: 10,
      onProgress: (done, total) => sp.stop(`Found ${done} uncertain examples`)
    })

    if (!result.uncertain.length) {
      warn('No uncertain examples found.')
      return
    }

    info(`\nMost uncertain examples (${result.uncertain.length}):`)
    table(
      result.uncertain.map((u, i) => [
        i + 1,
        u.text?.slice(0, 50) + (u.text?.length > 50 ? '...' : ''),
        u.predicted || '?',
        u.confidence !== null ? `${(u.confidence * 100).toFixed(0)}%` : '?'
      ]),
      ['#', 'Text', 'Predicted', 'Confidence']
    )

    // Offer labeling options
    const labelAction = await menu('What to do with uncertain examples?', [
      'LLM-in-the-loop (send to Claude for labeling)',
      'Skip (review only)',
    ])

    if (labelAction === 0) {
      const labelSp = spinner('Sending to Claude for labeling...')
      const labeled = await llmLabel(result.uncertain, task, {
        apiKey: API_KEY,
        model: task.synthetic?.model
      })
      labelSp.stop(`${labeled.length} examples labeled`)

      // Show labeling results
      table(
        labeled.map((l, i) => [
          i + 1,
          l.text?.slice(0, 40) + (l.text?.length > 40 ? '...' : ''),
          result.uncertain[i]?.predicted || '?',
          l.label
        ]),
        ['#', 'Text', 'Model said', 'Claude said']
      )

      // Offer to add to training data
      const addAction = await menu('Add these labeled examples to training data?', ['Yes, add to synthetic data', 'No, discard'])
      if (addAction === 0) {
        const { readJsonl, writeJsonl } = await import('./lib/data.js')
        const { join } = await import('node:path')
        const synPath = join('data', `${task.name}_synthetic.jsonl`)
        let existing = []
        try { existing = await readJsonl(synPath) } catch {}
        const newData = labeled.map(l => ({ text: l.text, label: l.label, _source: 'active_learning' }))
        await writeJsonl(synPath, [...existing, ...newData])
        success(`Added ${newData.length} examples to ${synPath}`)

        // Save iteration record
        await saveIteration(task.name, {
          examples_added: newData.length,
          accuracy_before: meta.accuracy,
          method: 'llm_labeling'
        })
      }
    }
  } catch (e) {
    sp.fail('Uncertainty sampling failed')
    error(e.message)
  }
}

async function runActiveLearningHistory(task) {
  header(`Active learning history for "${task.name}"`)

  const history = await loadHistory(task.name)
  if (!history.iterations.length) {
    warn('No active learning iterations yet.')
    return
  }

  table(
    history.iterations.map((it, i) => [
      i + 1,
      it.timestamp?.slice(0, 19) || 'N/A',
      it.method || 'unknown',
      it.examples_added || 0,
      it.accuracy_before?.toFixed(4) || 'N/A'
    ]),
    ['#', 'Timestamp', 'Method', 'Added', 'Accuracy before']
  )
}

async function runEmbedTrain(task) {
  header(`Train with embeddings for "${task.name}"`)

  const providers = listEmbedProviders().filter(p => p.configured)
  if (!providers.length) {
    warn('No embedding providers configured.')
    dim('Set OPENAI_API_KEY or ensure Ollama is running.')
    return
  }

  const provIdx = await menu('Embedding provider:', providers.map(p => `${p.name} (${p.defaultModel})`))
  const provider = providers[provIdx]

  // Dimensionality reduction option
  const reduceIdx = await menu('Dimensionality reduction:', ['None', 'PCA', 'SVD (truncated)'])
  const dimReduce = reduceIdx === 0 ? null : reduceIdx === 1 ? 'pca' : 'svd'
  let nComponents = 50
  if (dimReduce) {
    const nStr = await prompt('Target dimensions [50]:')
    nComponents = parseInt(nStr) || 50
  }

  // Prepare data first
  const splitResult = await runPrepare(task)
  if (!splitResult) return

  // Load the split data
  const { readJsonl, writeJsonl } = await import('./lib/data.js')
  const trainData = await readJsonl(splitResult.train.path)
  const valData = await readJsonl(splitResult.val.path)

  // Embed training data
  const sp = spinner(`Embedding ${trainData.length + valData.length} texts with ${provider.name}...`)
  try {
    const allTexts = [...trainData.map(d => d.text), ...valData.map(d => d.text)]
    const allEmbeddings = await cachedEmbed(provider.key, allTexts, {
      model: provider.defaultModel,
      onProgress: (done, total) => sp.stop(`Embedded ${done}/${total}`)
    })

    const trainEmbeddings = allEmbeddings.slice(0, trainData.length)
    const valEmbeddings = allEmbeddings.slice(trainData.length)

    sp.stop(`Embedded ${allTexts.length} texts (${trainEmbeddings[0].length} dimensions)`)

    // Write embeddings to JSONL for the Python training script
    const { join } = await import('node:path')
    const trainEmbPath = join('data', `${task.name}_train_embeddings.jsonl`)
    const valEmbPath = join('data', `${task.name}_val_embeddings.jsonl`)
    await writeJsonl(trainEmbPath, trainEmbeddings.map(e => ({ embedding: e })))
    await writeJsonl(valEmbPath, valEmbeddings.map(e => ({ embedding: e })))

    info(`Embeddings saved: ${trainEmbPath}, ${valEmbPath}`)

    // Run training with embeddings
    await runTrain(task, splitResult, {
      trainEmbeddings: trainEmbPath,
      valEmbeddings: valEmbPath,
      dimReduce,
      nComponents
    })
  } catch (e) {
    sp.fail('Embedding failed')
    error(e.message)
  }
}

async function runSemanticDedup(task) {
  header(`Semantic deduplication for "${task.name}"`)

  const providers = listEmbedProviders().filter(p => p.configured)
  if (!providers.length) {
    warn('No embedding providers configured.')
    return
  }

  const data = await loadAndMerge(task)
  if (!data.length) { warn('No data found.'); return }

  const provIdx = await menu('Embedding provider:', providers.map(p => `${p.name} (${p.defaultModel})`))
  const provider = providers[provIdx]

  const thresholdStr = await prompt('Similarity threshold (0.0-1.0) [0.92]:')
  const threshold = parseFloat(thresholdStr) || 0.92

  const sp = spinner(`Embedding ${data.length} texts...`)
  try {
    const texts = data.map(d => d.text)
    const embeddings = await cachedEmbed(provider.key, texts, {
      model: provider.defaultModel,
      onProgress: (done, total) => sp.stop(`Embedded ${done}/${total}`)
    })

    sp.stop(`Embedded ${data.length} texts, finding semantic duplicates...`)

    const result = await semanticDeduplicate(data, embeddings, { threshold })
    success(`Removed ${result.removed} semantic duplicates (${data.length} → ${result.data.length})`)
  } catch (e) {
    sp.fail('Semantic dedup failed')
    error(e.message)
  }
}

async function runEmbedCacheStats() {
  header('Embedding cache stats')

  try {
    const { join } = await import('node:path')
    const cachePath = join('data', 'embed_cache.sqlite')
    const file = Bun.file(cachePath)
    if (!await file.exists()) {
      warn('No embedding cache found yet.')
      return
    }

    const cache = createEmbedCache(cachePath)
    const providers = listEmbedProviders()
    const rows = []
    for (const p of providers) {
      const count = cache.count(p.defaultModel)
      if (count > 0) rows.push([p.name, p.defaultModel, count])
    }
    cache.close()

    if (rows.length) {
      table(rows, ['Provider', 'Model', 'Cached'])
    } else {
      info('Cache is empty.')
    }
  } catch (e) {
    error(e.message)
  }
}

// ── Phase 8: Curriculum & Data Strategy ──────────────────

async function runCurriculumAnalysis(task) {
  header(`Curriculum analysis for "${task.name}"`)
  const meta = await loadMeta(task.name)
  if (!meta) { warn('No trained model found. Train one first.'); return }

  const data = await loadAndMerge(task)
  if (!data.length) { warn('No data found.'); return }

  const sp = spinner('Scoring example difficulty...')
  try {
    const scored = await scoreDifficulty(task.name, data)
    const stages = curriculumStages(scored)
    sp.stop(`Scored ${scored.length} examples`)

    table([
      ['Easy (confident)', stages.easy.length],
      ['Medium', stages.medium.length],
      ['Hard (uncertain)', stages.hard.length],
      ['Misclassified', scored.filter(d => !d._correct).length]
    ], ['Stage', 'Count'])
  } catch (e) {
    sp.fail('Scoring failed')
    error(e.message)
  }
}

async function runLLMJudge(task) {
  if (!API_KEY) { error('Set ANTHROPIC_API_KEY in your environment'); return }
  header(`LLM-as-judge scoring for "${task.name}"`)

  const data = await loadAndMerge(task)
  if (!data.length) { warn('No data found.'); return }

  const thresholdStr = await prompt('Quality threshold (0.0-1.0) [0.7]:')
  const threshold = parseFloat(thresholdStr) || 0.7

  const sp = spinner(`Scoring ${data.length} examples...`)
  const scored = await llmJudge(data, task, {
    apiKey: API_KEY,
    onProgress: (done, total) => sp.stop(`Scored ${done}/${total}`)
  })
  sp.stop(`Scored ${scored.length} examples`)

  const result = filterByQuality(scored, threshold)
  success(`Kept ${result.data.length}/${scored.length} examples (${result.removed} below threshold ${threshold})`)

  // Show quality distribution
  const high = scored.filter(d => d._quality >= 0.8).length
  const mid = scored.filter(d => d._quality >= 0.5 && d._quality < 0.8).length
  const low = scored.filter(d => d._quality < 0.5).length
  table([['High (≥0.8)', high], ['Medium (0.5-0.8)', mid], ['Low (<0.5)', low]], ['Quality', 'Count'])
}

async function runContrastive(task) {
  if (!API_KEY) { error('Set ANTHROPIC_API_KEY in your environment'); return }
  if (task.type !== 'classification') { warn('Contrastive generation requires a classification task'); return }

  header(`Contrastive generation for "${task.name}"`)
  const countStr = await prompt('Examples per label pair [5]:')
  const count = parseInt(countStr) || 5

  const sp = spinner('Generating contrastive examples...')
  try {
    const examples = await generateContrastive(task, {
      apiKey: API_KEY,
      count,
      onProgress: (done, total) => sp.stop(`Generated pair ${done}/${total}`)
    })
    sp.stop(`Generated ${examples.length} contrastive examples`)

    if (examples.length) {
      table(
        examples.slice(0, 10).map((e, i) => [i + 1, e.text?.slice(0, 40) + '...', e.label, e._confused_with || '?']),
        ['#', 'Text', 'Label', 'Confused with']
      )
    }
  } catch (e) {
    sp.fail('Generation failed')
    error(e.message)
  }
}

async function runEnsembleGenerate(task) {
  if (!API_KEY) { error('Set ANTHROPIC_API_KEY in your environment'); return }
  header(`Cross-provider ensemble for "${task.name}"`)

  const available = listProviders().filter(p => p.configured)
  if (available.length < 2) { warn('Need at least 2 configured providers.'); return }

  info(`Available: ${available.map(p => p.name).join(', ')}`)
  const countStr = await prompt('Total examples to generate [100]:')
  const count = parseInt(countStr) || 100

  const sp = spinner('Generating from multiple providers...')
  try {
    const all = await ensembleGenerate(task, available, {
      apiKey: API_KEY,
      count,
      onProgress: (done, total) => sp.stop(`Generated ${done}/${total}`)
    })
    sp.stop(`Generated ${all.length} examples from ${available.length} providers`)

    const byProvider = {}
    for (const row of all) {
      const p = row._provider || 'unknown'
      byProvider[p] = (byProvider[p] || 0) + 1
    }
    table(Object.entries(byProvider).map(([k, v]) => [k, v]), ['Provider', 'Count'])
  } catch (e) {
    sp.fail('Ensemble generation failed')
    error(e.message)
  }
}

// ── Phase 9: Multi-task & Transfer Learning ──────────────

async function runZeroShot(task) {
  if (!API_KEY) { error('Set ANTHROPIC_API_KEY in your environment'); return }
  if (task.type !== 'classification') { warn('Zero-shot eval requires a classification task'); return }

  header(`Zero-shot evaluation for "${task.name}"`)

  const { readJsonl } = await import('./lib/data.js')
  const valPath = join('data', `${task.name}_val.jsonl`)
  let valData
  try { valData = await readJsonl(valPath) } catch { warn('No validation data. Run prepare first.'); return }

  const sp = spinner(`Classifying ${valData.length} examples with LLM...`)
  const result = await zeroShotEval(task, valData, {
    apiKey: API_KEY,
    onProgress: (done, total) => sp.stop(`Classified ${done}/${total}`)
  })
  sp.stop(`Zero-shot evaluation complete`)

  success(`Zero-shot accuracy: ${(result.accuracy * 100).toFixed(1)}% (${result.correct}/${result.total})`)

  const meta = await loadMeta(task.name)
  if (meta) {
    const delta = result.accuracy - meta.accuracy
    info(`Trained model accuracy: ${(meta.accuracy * 100).toFixed(1)}% (${delta > 0 ? 'LLM is better by' : 'distilled model wins by'} ${(Math.abs(delta) * 100).toFixed(1)}pp)`)
  }
}

// ── Phase 10: Evaluation & Interpretability ──────────────

async function runKFoldCV(task) {
  header(`K-fold cross-validation for "${task.name}"`)

  const data = await loadAndMerge(task)
  if (data.length < 10) { warn('Need at least 10 examples for cross-validation.'); return }

  const kStr = await prompt('Number of folds [5]:')
  const k = parseInt(kStr) || 5

  const algoIdx = await menu('Algorithm:', ['logistic_regression', 'svm', 'random_forest'])
  const algorithm = ['logistic_regression', 'svm', 'random_forest'][algoIdx]

  info(`Running ${k}-fold cross-validation with ${algorithm}...`)
  const result = await kFoldCV(task, data, {
    k,
    algorithm,
    onFold: (fold, total, acc) => info(`  Fold ${fold}/${total}: ${(acc * 100).toFixed(1)}%`)
  })

  success(`Mean accuracy: ${(result.mean * 100).toFixed(1)}% ± ${(result.std * 100).toFixed(1)}%`)
  table(
    result.scores.map((s, i) => [`Fold ${i + 1}`, (s * 100).toFixed(1) + '%']),
    ['Fold', 'Accuracy']
  )
}

async function runFeatureImportance(task) {
  header(`Feature importance for "${task.name}"`)

  const meta = await loadMeta(task.name)
  if (!meta) { warn('No trained model found.'); return }

  const sp = spinner('Extracting features...')
  try {
    const importance = await featureImportance(task.name)
    sp.stop('Done')

    if (importance.error) { warn(importance.error); return }

    for (const [label, features] of Object.entries(importance)) {
      info(`\n  Label: ${label}`)
      table(
        features.slice(0, 10).map((f, i) => [i + 1, f.feature, f.weight.toFixed(4)]),
        ['#', 'Feature', 'Weight']
      )
    }
  } catch (e) {
    sp.fail('Failed')
    error(e.message)
  }
}

async function runErrorTaxonomy(task) {
  header(`Error taxonomy for "${task.name}"`)

  const meta = await loadMeta(task.name)
  if (!meta) { warn('No trained model found.'); return }

  const { readJsonl } = await import('./lib/data.js')
  const valPath = join('data', `${task.name}_val.jsonl`)
  let valData
  try { valData = await readJsonl(valPath) } catch { warn('No validation data.'); return }

  const sp = spinner('Analyzing errors...')
  const predictions = await predict(task.name, valData.map(d => d.text))
  const taxonomy = errorTaxonomy(valData, predictions)
  sp.stop(`${taxonomy.totalErrors} errors / ${taxonomy.totalErrors + taxonomy.totalCorrect} total`)

  if (taxonomy.topConfusions.length) {
    info('\nTop confusion pairs:')
    table(
      taxonomy.topConfusions.map(c => [c.pair, c.count]),
      ['Confusion', 'Count']
    )
  }

  info('\nErrors by text length:')
  table(
    [['Short (<50 chars)', taxonomy.byTextLength.short], ['Medium (50-150)', taxonomy.byTextLength.medium], ['Long (>150)', taxonomy.byTextLength.long]],
    ['Length', 'Errors']
  )
}

async function runCalibration(task) {
  header(`Calibration analysis for "${task.name}"`)

  const meta = await loadMeta(task.name)
  if (!meta) { warn('No trained model found.'); return }

  const { readJsonl } = await import('./lib/data.js')
  const valPath = join('data', `${task.name}_val.jsonl`)
  let valData
  try { valData = await readJsonl(valPath) } catch { warn('No validation data.'); return }

  const sp = spinner('Running calibration analysis...')
  const predictions = await predict(task.name, valData.map(d => d.text))
  const actual = valData.map(d => d.label)
  const cal = calibrationBins(predictions, actual)
  sp.stop('Done')

  if (!cal.hasConfidence) { warn('Model does not provide confidence scores (e.g. SVM). Try logistic regression.'); return }

  success(`Expected Calibration Error (ECE): ${(cal.ece * 100).toFixed(2)}%`)

  const binRows = cal.bins.map((b, i) => {
    const lo = (i / cal.bins.length * 100).toFixed(0)
    const hi = ((i + 1) / cal.bins.length * 100).toFixed(0)
    return [`${lo}-${hi}%`, b.predictions, b.correct, b.predictions > 0 ? (b.avgAccuracy * 100).toFixed(1) + '%' : '-']
  }).filter(r => r[1] > 0)

  table(binRows, ['Confidence', 'Predictions', 'Correct', 'Accuracy'])

  if (cal.ece < 0.05) info('Model is well-calibrated.')
  else if (cal.ece < 0.15) warn('Model is slightly miscalibrated.')
  else warn('Model is poorly calibrated — consider Platt scaling.')
}

// ── Phase 11: Ensemble & Experiments ──────────────────────

async function runTrainEnsemble(task) {
  header(`Train ensemble for "${task.name}"`)

  const splitResult = await runPrepare(task)
  if (!splitResult) return

  const sp = spinner('Training all algorithms...')
  try {
    const result = await trainEnsembleModels(task, splitResult.train.path, splitResult.val.path, {
      onAlgorithm: algo => sp.stop(`Training ${algo}...`),
      onStdout: () => {},
      onStderr: () => {}
    })
    sp.stop(`Trained ${result.models.length} models`)

    table(
      result.models.map(m => [m.algorithm, (m.accuracy * 100).toFixed(1) + '%']),
      ['Algorithm', 'Accuracy']
    )

    // Record each as an experiment
    const { readJsonl } = await import('./lib/data.js')
    const trainData = await readJsonl(splitResult.train.path)
    const dHash = hashDataset(trainData)
    for (const m of result.models) {
      recordExperiment({
        task: task.name,
        algorithm: m.algorithm,
        accuracy: m.accuracy,
        trainSize: splitResult.train.count,
        valSize: splitResult.val.count,
        dataHash: dHash,
        featureMode: 'tfidf',
        labels: task.labels || null,
        notes: 'ensemble training'
      })
    }
    dim('  Experiments recorded.')
  } catch (e) {
    sp.fail('Ensemble training failed')
    error(e.message)
  }
}

async function runEnsemblePredict(task) {
  header(`Ensemble predict for "${task.name}"`)

  const models = await listEnsembleModels(task.name)
  if (!models.length) {
    warn('No ensemble models found. Run "Train ensemble" first.')
    return
  }

  info(`Ensemble: ${models.map(m => m.algorithm).join(', ')}`)

  const thresholdStr = await prompt('Rejection threshold (0 = accept all) [0]:')
  const threshold = parseFloat(thresholdStr) || 0

  while (true) {
    const text = await prompt('Enter text (or "back" to return):')
    if (text.toLowerCase() === 'back') return

    try {
      const results = await ensemblePredict(task.name, [text], { threshold })
      const r = results[0]
      if (r.rejected) {
        warn(`Rejected (confidence: ${(r.confidence * 100).toFixed(1)}% < threshold ${(threshold * 100).toFixed(1)}%)`)
        dim(`  Votes: ${JSON.stringify(r.votes)}`)
      } else {
        success(`Label: ${r.label}  (confidence: ${(r.confidence * 100).toFixed(1)}%, agreement: ${(r.agreement * 100).toFixed(0)}%)`)
        dim(`  Per model: ${r.perModel.map(m => `${m.algorithm}=${m.label}`).join(', ')}`)
      }
    } catch (e) {
      error(e.message)
    }
  }
}

async function runThresholdPredict(task) {
  header(`Predict with confidence threshold for "${task.name}"`)

  const meta = await loadMeta(task.name)
  if (!meta) { warn('No trained model found.'); return }

  const thresholdStr = await prompt('Rejection threshold [0.5]:')
  const threshold = parseFloat(thresholdStr) || 0.5
  info(`Rejecting predictions below ${(threshold * 100).toFixed(0)}% confidence`)

  while (true) {
    const text = await prompt('Enter text (or "back" to return):')
    if (text.toLowerCase() === 'back') return

    try {
      const results = await predictWithThreshold(task.name, [text], { threshold })
      const r = results[0]
      if (r.rejected) {
        warn(`Rejected: ${r.label} (confidence: ${(r.confidence * 100).toFixed(1)}% — below threshold)`)
      } else {
        success(`Label: ${r.label}  (confidence: ${((r.confidence || 0) * 100).toFixed(1)}%)`)
      }
    } catch (e) {
      error(e.message)
    }
  }
}

async function runExperimentHistory(task) {
  header(`Experiment history for "${task.name}"`)

  const experiments = listExperiments(task.name)
  if (!experiments.length) {
    warn('No experiments recorded yet. Train a model to start tracking.')
    return
  }

  const stats = experimentStats(task.name)
  info(`${stats.total} experiments | ${stats.algorithms_tried} algorithms | ${stats.data_versions} data versions`)
  info(`Best: ${(stats.best_accuracy * 100).toFixed(1)}% | Avg: ${(stats.avg_accuracy * 100).toFixed(1)}% | Worst: ${(stats.worst_accuracy * 100).toFixed(1)}%`)

  table(
    experiments.slice(0, 20).map(e => [
      e.id,
      e.timestamp?.slice(0, 19),
      e.algorithm || '?',
      e.accuracy != null ? (e.accuracy * 100).toFixed(1) + '%' : '-',
      e.train_size || '-',
      e.feature_mode || 'tfidf',
      e.data_hash?.slice(0, 8) || '-'
    ]),
    ['ID', 'Timestamp', 'Algorithm', 'Accuracy', 'Train', 'Features', 'Data hash']
  )

  // Offer comparison
  if (experiments.length >= 2) {
    const compareChoice = await menu('Compare experiments?', ['Yes', 'No'])
    if (compareChoice === 0) {
      const idA = await prompt(`First experiment ID [${experiments[1]?.id}]:`)
      const idB = await prompt(`Second experiment ID [${experiments[0]?.id}]:`)
      const a = parseInt(idA) || experiments[1]?.id
      const b = parseInt(idB) || experiments[0]?.id

      try {
        const cmp = compareExperiments(a, b)
        info(`\nComparing #${a} vs #${b}:`)
        const delta = cmp.diff.accuracyDelta
        if (delta > 0) success(`Accuracy: +${(delta * 100).toFixed(2)}pp improvement`)
        else if (delta < 0) warn(`Accuracy: ${(delta * 100).toFixed(2)}pp regression`)
        else info('Accuracy: unchanged')

        if (cmp.diff.changes.length) {
          info('Changes:')
          for (const c of cmp.diff.changes) dim(`  • ${c}`)
        }
      } catch (e) {
        error(e.message)
      }
    }
  }
}

// ── Phase 12: Transformer Distillation ───────────────────

async function runTransformerTrain(task) {
  header(`Transformer fine-tuning for "${task.name}"`)

  // Check dependencies first
  const deps = await checkDeps()
  if (!deps.torch || !deps.transformers) {
    warn('Transformer training requires PyTorch and HuggingFace Transformers.')
    if (!deps.torch) info('  Install PyTorch: pip3 install torch')
    if (!deps.transformers) info('  Install Transformers: pip3 install transformers')
    return
  }

  // Show device info
  const device = await detectDevice()
  info(`Compute device: ${device.device} (${device.info})`)

  // Prepare data
  const splitResult = await runPrepare(task)
  if (!splitResult) return

  // Model selection
  const presets = listModelPresets()
  const modelIdx = await menu('Choose model:', [
    ...presets.map(p => `${p.key} (${p.params}) — ${p.description}`),
    'Custom HuggingFace model'
  ])

  let model
  if (modelIdx < presets.length) {
    model = presets[modelIdx].key
  } else {
    model = await prompt('HuggingFace model name (e.g. bert-base-uncased):')
    if (!model) { warn('No model specified.'); return }
  }

  // Training config
  const epochsStr = await prompt('Epochs [default for model]:')
  const epochs = parseInt(epochsStr) || undefined
  const bsStr = await prompt('Batch size [16]:')
  const batchSize = parseInt(bsStr) || 16

  const onnxIdx = await menu('Export to ONNX?', ['No', 'Yes'])
  const onnx = onnxIdx === 1

  // Version existing model
  const meta = await loadMeta(task.name)
  if (meta) {
    try {
      const v = await versionModel(task.name)
      dim(`  Previous model versioned: ${v.version}`)
    } catch {}
  }

  const sp = spinner('Fine-tuning transformer...')
  const startTime = Date.now()
  try {
    const result = await trainTransformer(task, splitResult.train.path, splitResult.val.path, {
      model,
      epochs,
      batchSize,
      onnx,
      onEpoch: (epoch, total) => sp.stop(`Epoch ${epoch}/${total}...`),
      onEvalLog: data => {
        if (data.accuracy) info(`  Epoch ${data.epoch} accuracy: ${(data.accuracy * 100).toFixed(1)}%`)
      },
      onStdout: line => {
        if (line.includes('Validation Accuracy')) sp.stop(line.trim())
      },
      onStderr: text => {} // suppress HF warnings
    })
    const durationMs = Date.now() - startTime
    sp.stop(`Training complete`)
    success(`Accuracy: ${(result.accuracy * 100).toFixed(1)}% | Model: ${result.model} | Device: ${result.device} | Time: ${result.duration}s`)
    success(`Model saved to ${result.modelDir}`)

    // Record experiment
    try {
      const { readJsonl } = await import('./lib/data.js')
      const trainData = await readJsonl(splitResult.train.path)
      const valData = await readJsonl(splitResult.val.path)
      recordExperiment({
        task: task.name,
        algorithm: `transformer:${model}`,
        accuracy: result.accuracy,
        trainSize: trainData.length,
        valSize: valData.length,
        dataHash: hashDataset(trainData),
        featureMode: 'transformer',
        hyperparams: { epochs, batchSize, model },
        labels: task.labels || null,
        durationMs
      })
      dim('  Experiment recorded.')
    } catch {}
  } catch (e) {
    sp.fail('Training failed')
    error(e.message)
  }
}

async function runTransformerPredict(task) {
  header(`Transformer predict for "${task.name}"`)

  const meta = await loadMeta(task.name)
  if (!meta || meta.feature_mode !== 'transformer') {
    warn('No transformer model found. Run "Train transformer" first.')
    return
  }

  info(`Model: ${meta.model_name || meta.algorithm} | Accuracy: ${meta.accuracy?.toFixed(4)}`)

  while (true) {
    const text = await prompt('Enter text (or "back" to return):')
    if (text.toLowerCase() === 'back') return

    try {
      const results = await predictTransformer(task.name, [text], { maxLength: meta.max_length })
      const r = results[0]
      success(`Label: ${r.label}  (confidence: ${(r.confidence * 100).toFixed(1)}%)`)
    } catch (e) {
      error(`Prediction failed: ${e.message}`)
    }
  }
}

async function runTransformerCompare(task) {
  header(`Compare transformer vs classical models for "${task.name}"`)

  const meta = await loadMeta(task.name)
  if (!meta) { warn('No trained model found. Train at least one model first.'); return }

  const experiments = listExperiments(task.name, { limit: 20 })
  if (experiments.length < 1) { warn('No experiments recorded. Train some models first.'); return }

  info('Recent experiments:')
  table(
    experiments.slice(0, 10).map(e => [
      e.id,
      e.algorithm || '?',
      e.accuracy != null ? (e.accuracy * 100).toFixed(1) + '%' : '-',
      e.feature_mode || '?',
      e.duration_ms ? `${(e.duration_ms / 1000).toFixed(1)}s` : '-'
    ]),
    ['ID', 'Algorithm', 'Accuracy', 'Features', 'Duration']
  )
}

// ── CRF Sequence Labeling ────────────────────────────────

async function runCRFTrain(task) {
  if (task.type !== 'sequence-labeling') {
    warn('CRF training requires a sequence-labeling task'); return
  }

  header(`Train CRF for "${task.name}"`)
  const tags = labelsToBIO(task.labels)
  info(`Entity types: ${task.labels.join(', ')}`)
  info(`BIO tags: ${tags.join(', ')}`)

  // Load data
  const { readJsonl } = await import('./lib/data.js')
  const dataPath = join('data', `${task.name}_synthetic.jsonl`)
  const file = Bun.file(dataPath)
  if (!await file.exists()) { warn('No data found. Generate data first.'); return }

  const data = await readJsonl(dataPath)
  const seqData = data.filter(d => Array.isArray(d.tokens) && Array.isArray(d.tags))
  if (seqData.length < 2) { warn(`Only ${seqData.length} valid sequences found. Need at least 2.`); return }

  // Split
  const splitIdx = Math.floor(seqData.length * 0.8)
  const trainData = seqData.slice(0, splitIdx)
  const valData = seqData.slice(splitIdx)
  info(`Train: ${trainData.length} sequences, Val: ${valData.length} sequences`)

  // Train
  const epochsStr = await prompt('Epochs [10]:')
  const epochs = parseInt(epochsStr) || 10

  const startTime = Date.now()
  const model = trainCRF(trainData, {
    tags,
    epochs,
    onEpoch: ({ epoch, epochs, accuracy, sequencesCorrect, sequencesTotal }) => {
      progress(epoch, epochs, `epoch — ${sequencesCorrect}/${sequencesTotal} sequences correct (${(accuracy * 100).toFixed(1)}%)`)
    }
  })

  const duration = Date.now() - startTime
  success(`Training complete in ${(duration / 1000).toFixed(1)}s`)

  // Evaluate
  const predictions = predictBatch(valData.map(d => d.tokens), model)
  const eval_ = evaluateEntities(valData, predictions)

  info(`Token accuracy: ${(eval_.tokenAccuracy * 100).toFixed(1)}%`)
  info(`Entity F1 (micro): ${(eval_.micro.f1 * 100).toFixed(1)}%`)
  info(`Entities — gold: ${eval_.totalEntities.gold}, predicted: ${eval_.totalEntities.predicted}`)

  if (Object.keys(eval_.byType).length > 0) {
    table(
      Object.entries(eval_.byType).map(([type, m]) => [
        type,
        (m.precision * 100).toFixed(1) + '%',
        (m.recall * 100).toFixed(1) + '%',
        (m.f1 * 100).toFixed(1) + '%',
        m.support
      ]),
      ['Entity', 'Precision', 'Recall', 'F1', 'Support']
    )
  }

  // Save
  const modelDir = await saveCRFModel(task.name, model)
  success(`Model saved to ${modelDir}`)

  // Record experiment
  try {
    recordExperiment({
      task: task.name,
      algorithm: 'crf',
      accuracy: eval_.micro.f1,
      train_size: trainData.length,
      val_size: valData.length,
      feature_mode: 'crf',
      hyperparams: { epochs, hashSize: model.hashSize },
      duration_ms: duration,
      labels: task.labels
    })
    dim('  Experiment recorded')
  } catch {}
}

async function runCRFPredict(task) {
  if (task.type !== 'sequence-labeling') {
    warn('CRF prediction requires a sequence-labeling task'); return
  }

  const model = await loadCRFModel(task.name)
  if (!model) { warn('No CRF model found. Train one first.'); return }

  header(`CRF Predict — "${task.name}"`)
  info('Enter text to tag (or "q" to quit):')

  while (true) {
    const text = await prompt('Text:')
    if (text === 'q' || text === 'quit') break

    const tokens = text.split(/\s+/)
    const tags = predictSequence(tokens, model)
    const entities = extractEntities(tokens, tags)

    // Show tagged output
    const tagged = tokens.map((t, i) => {
      if (tags[i] === 'O') return t
      return `[${t}/${tags[i]}]`
    }).join(' ')
    info(tagged)

    if (entities.length > 0) {
      for (const e of entities) {
        dim(`  ${e.type}: "${e.text}"`)
      }
    } else {
      dim('  (no entities found)')
    }
  }
}

async function runCRFEval(task) {
  if (task.type !== 'sequence-labeling') {
    warn('CRF evaluation requires a sequence-labeling task'); return
  }

  const model = await loadCRFModel(task.name)
  if (!model) { warn('No CRF model found. Train one first.'); return }

  header(`CRF Evaluation — "${task.name}"`)

  const { readJsonl } = await import('./lib/data.js')
  const dataPath = join('data', `${task.name}_synthetic.jsonl`)
  const file = Bun.file(dataPath)
  if (!await file.exists()) { warn('No data found.'); return }

  const data = await readJsonl(dataPath)
  const seqData = data.filter(d => Array.isArray(d.tokens) && Array.isArray(d.tags))
  const valData = seqData.slice(Math.floor(seqData.length * 0.8))

  if (valData.length === 0) { warn('No validation data.'); return }

  const predictions = predictBatch(valData.map(d => d.tokens), model)
  const eval_ = evaluateEntities(valData, predictions)

  info(`Sequences: ${valData.length}`)
  info(`Token accuracy: ${(eval_.tokenAccuracy * 100).toFixed(1)}%`)
  info(`Entity F1 (micro): P=${(eval_.micro.precision * 100).toFixed(1)}% R=${(eval_.micro.recall * 100).toFixed(1)}% F1=${(eval_.micro.f1 * 100).toFixed(1)}%`)

  if (Object.keys(eval_.byType).length > 0) {
    table(
      Object.entries(eval_.byType).map(([type, m]) => [
        type,
        (m.precision * 100).toFixed(1) + '%',
        (m.recall * 100).toFixed(1) + '%',
        (m.f1 * 100).toFixed(1) + '%',
        m.support
      ]),
      ['Entity', 'Precision', 'Recall', 'F1', 'Support']
    )
  }

  // Show sample predictions
  info('Sample predictions:')
  for (let i = 0; i < Math.min(3, valData.length); i++) {
    const gold = valData[i]
    const pred = predictions[i]
    dim(`  Gold: ${gold.tokens.map((t, j) => gold.tags[j] !== 'O' ? `[${t}/${gold.tags[j]}]` : t).join(' ')}`)
    dim(`  Pred: ${pred.tokens.map((t, j) => pred.tags[j] !== 'O' ? `[${t}/${pred.tags[j]}]` : t).join(' ')}`)
    dim('')
  }
}

// ── Phase 15: Scoring Tasks ──────────────────────────────

async function runScoringTrain(task) {
  if (task.type !== 'scoring' && task.type !== 'regression') {
    warn('Scoring training requires a scoring or regression task'); return
  }

  header(`Train scoring model for "${task.name}"`)
  const range = task.scoreRange || { min: 0, max: 5 }
  info(`Score range: ${range.min} to ${range.max}`)

  // Load data
  const { readJsonl } = await import('./lib/data.js')
  const dataPath = join('data', `${task.name}_synthetic.jsonl`)
  const file = Bun.file(dataPath)
  if (!await file.exists()) { warn('No data found. Generate data first.'); return }

  const data = await readJsonl(dataPath)
  const scoringData = data.filter(d => typeof d.text === 'string' && typeof d.value === 'number' && !isNaN(d.value))
  if (scoringData.length < 2) { warn(`Only ${scoringData.length} valid examples found. Need at least 2.`); return }

  // Split
  const splitIdx = Math.floor(scoringData.length * 0.8)
  const trainData = scoringData.slice(0, splitIdx)
  const valData = scoringData.slice(splitIdx)
  info(`Train: ${trainData.length} examples, Val: ${valData.length} examples`)

  // Training config
  const epochsStr = await prompt('Epochs [20]:')
  const epochs = parseInt(epochsStr) || 20

  const startTime = Date.now()
  const model = trainScoring(trainData, {
    epochs,
    minVal: range.min,
    maxVal: range.max,
    onEpoch: ({ epoch, epochs: total, mse }) => {
      progress(epoch, total, `epoch — MSE: ${mse.toFixed(4)}`)
    }
  })

  const duration = Date.now() - startTime
  success(`Training complete in ${(duration / 1000).toFixed(1)}s`)

  // Evaluate
  const predictions = predictScoreBatch(valData.map(d => d.text), model)
  const eval_ = evaluateScoring(valData, predictions)

  info(`MSE: ${eval_.mse.toFixed(4)} | MAE: ${eval_.mae.toFixed(4)} | RMSE: ${eval_.rmse.toFixed(4)}`)
  info(`Correlation: ${eval_.correlation.toFixed(4)} | R\u00b2: ${eval_.r2.toFixed(4)}`)

  // Show sample predictions
  info('Sample predictions:')
  const sampleCount = Math.min(5, valData.length)
  table(
    valData.slice(0, sampleCount).map((d, i) => [
      d.text.slice(0, 50) + (d.text.length > 50 ? '...' : ''),
      d.value.toFixed(2),
      predictions[i].score.toFixed(2),
      Math.abs(d.value - predictions[i].score).toFixed(2)
    ]),
    ['Text', 'Gold', 'Predicted', 'Error']
  )

  // Save
  const modelDir = await saveScoringModel(task.name, model)
  success(`Model saved to ${modelDir}`)

  // Record experiment
  try {
    recordExperiment({
      task: task.name,
      algorithm: 'scoring',
      accuracy: eval_.correlation,
      train_size: trainData.length,
      val_size: valData.length,
      feature_mode: 'scoring',
      hyperparams: { epochs, hashSize: model.hashSize },
      duration_ms: duration,
      notes: `MSE=${eval_.mse.toFixed(4)} MAE=${eval_.mae.toFixed(4)} R2=${eval_.r2.toFixed(4)}`
    })
    dim('  Experiment recorded')
  } catch {}
}

async function runScoringPredict(task) {
  if (task.type !== 'scoring' && task.type !== 'regression') {
    warn('Scoring prediction requires a scoring or regression task'); return
  }

  const model = await loadScoringModel(task.name)
  if (!model) { warn('No scoring model found. Train one first.'); return }

  header(`Scoring predict — "${task.name}"`)
  const range = task.scoreRange || { min: model.minVal, max: model.maxVal }
  info(`Score range: ${range.min} to ${range.max}`)

  while (true) {
    const text = await prompt('Enter text (or "back" to return):')
    if (text.toLowerCase() === 'back') return

    const score = predictScore(text, model)
    success(`Score: ${score.toFixed(2)}`)
  }
}

async function runScoringEval(task) {
  if (task.type !== 'scoring' && task.type !== 'regression') {
    warn('Scoring evaluation requires a scoring or regression task'); return
  }

  const model = await loadScoringModel(task.name)
  if (!model) { warn('No scoring model found. Train one first.'); return }

  header(`Scoring evaluation — "${task.name}"`)

  const { readJsonl } = await import('./lib/data.js')
  const dataPath = join('data', `${task.name}_synthetic.jsonl`)
  const file = Bun.file(dataPath)
  if (!await file.exists()) { warn('No data found.'); return }

  const data = await readJsonl(dataPath)
  const scoringData = data.filter(d => typeof d.text === 'string' && typeof d.value === 'number')
  const valData = scoringData.slice(Math.floor(scoringData.length * 0.8))

  if (valData.length === 0) { warn('No validation data.'); return }

  const predictions = predictScoreBatch(valData.map(d => d.text), model)
  const eval_ = evaluateScoring(valData, predictions)

  info(`Examples: ${eval_.n}`)
  table([
    ['MSE', eval_.mse.toFixed(4)],
    ['MAE', eval_.mae.toFixed(4)],
    ['RMSE', eval_.rmse.toFixed(4)],
    ['Correlation', eval_.correlation.toFixed(4)],
    ['R\u00b2', eval_.r2.toFixed(4)]
  ], ['Metric', 'Value'])

  // Error distribution
  const errors = valData.map((d, i) => Math.abs(d.value - predictions[i].score))
  const small = errors.filter(e => e < 0.5).length
  const medium = errors.filter(e => e >= 0.5 && e < 1.0).length
  const large = errors.filter(e => e >= 1.0).length
  info('\nError distribution:')
  table([
    ['< 0.5', small, `${(small / errors.length * 100).toFixed(0)}%`],
    ['0.5 - 1.0', medium, `${(medium / errors.length * 100).toFixed(0)}%`],
    ['> 1.0', large, `${(large / errors.length * 100).toFixed(0)}%`]
  ], ['Error', 'Count', '%'])

  // Show worst predictions
  info('\nWorst predictions:')
  const indexed = valData.map((d, i) => ({ ...d, pred: predictions[i].score, err: Math.abs(d.value - predictions[i].score) }))
  indexed.sort((a, b) => b.err - a.err)
  table(
    indexed.slice(0, 5).map(d => [
      d.text.slice(0, 40) + (d.text.length > 40 ? '...' : ''),
      d.value.toFixed(2),
      d.pred.toFixed(2),
      d.err.toFixed(2)
    ]),
    ['Text', 'Gold', 'Predicted', 'Error']
  )
}

// ── Phase 16: Active Learning Loop ───────────────────────

async function runActiveLoop(task) {
  if (!API_KEY) { error('Set ANTHROPIC_API_KEY in your environment'); return }

  // Check if model exists for this task type
  if (task.type === 'scoring' || task.type === 'regression') {
    if (!await hasScoringModel(task.name)) { warn('No scoring model found. Train one first.'); return }
  } else if (task.type === 'sequence-labeling') {
    if (!await hasCRFModel(task.name)) { warn('No CRF model found. Train one first.'); return }
  } else {
    const meta = await loadMeta(task.name)
    if (!meta) { warn('No trained model found. Train one first.'); return }
  }

  header(`Active learning loop — "${task.name}" (${task.type})`)

  const poolStr = await prompt('Pool size (candidates to generate) [30]:')
  const poolSize = parseInt(poolStr) || 30
  const selectStr = await prompt('Select top-K most uncertain [10]:')
  const selectK = parseInt(selectStr) || 10

  let sp = spinner('Generating candidate pool...')
  try {
    const result = await activeLoop(task, {
      apiKey: API_KEY,
      poolSize,
      selectK,
      onPool: count => sp.stop(`Generated ${count} candidates`),
      onUncertainty: count => { sp.stop(`Scored ${count} examples`); sp = spinner('Selecting most uncertain...') },
      onLabeled: count => sp.stop(`LLM labeled ${count} examples`)
    })

    if (!result.selected.length) {
      warn('No uncertain examples found.')
      return
    }

    // Show uncertain examples
    info(`\nMost uncertain examples (${result.selected.length}):`)
    if (task.type === 'scoring' || task.type === 'regression') {
      table(
        result.selected.map((s, i) => [
          i + 1,
          (s.text || '').slice(0, 45) + ((s.text || '').length > 45 ? '...' : ''),
          s.prediction != null ? s.prediction.toFixed(2) : '?',
          s.uncertainty.toFixed(4)
        ]),
        ['#', 'Text', 'Predicted', 'Uncertainty']
      )
    } else if (task.type === 'sequence-labeling') {
      table(
        result.selected.map((s, i) => [
          i + 1,
          (s.text || '').slice(0, 45) + ((s.text || '').length > 45 ? '...' : ''),
          s.margin != null ? s.margin.toFixed(2) : '?',
          s.uncertainty.toFixed(4)
        ]),
        ['#', 'Text', 'Margin', 'Uncertainty']
      )
    } else {
      table(
        result.selected.map((s, i) => [
          i + 1,
          (s.text || '').slice(0, 45) + ((s.text || '').length > 45 ? '...' : ''),
          s.prediction || '?',
          s.confidence != null ? `${(s.confidence * 100).toFixed(0)}%` : '?',
          s.uncertainty.toFixed(4)
        ]),
        ['#', 'Text', 'Predicted', 'Confidence', 'Uncertainty']
      )
    }

    // Show LLM-labeled results
    if (result.labeled.length) {
      info(`\nLLM labeled ${result.labeled.length} examples:`)
      for (const l of result.labeled.slice(0, 5)) {
        dim(`  ${JSON.stringify(l)}`)
      }
      if (result.labeled.length > 5) dim(`  ... and ${result.labeled.length - 5} more`)
    }

    // Offer to add to training data
    const addAction = await menu('Add labeled examples to training data?', ['Yes, add to synthetic data', 'No, discard'])
    if (addAction === 0) {
      const appendResult = await appendLabeledData(task.name, result.labeled, task.type)
      success(`Added ${appendResult.added} examples to ${appendResult.path} (${appendResult.total} total)`)

      // Save iteration
      await saveActiveIteration(task.name, {
        type: task.type,
        method: `active_loop_${task.type}`,
        pool_size: result.poolSize,
        selected: result.selected.length,
        labeled: result.labeled.length,
        added: appendResult.added
      })
      dim('  Iteration recorded')
    }
  } catch (e) {
    sp.fail('Active learning loop failed')
    error(e.message)
  }
}

async function runActiveLoopHistory(task) {
  header(`Active learning loop history — "${task.name}"`)

  const history = await loadActiveHistory(task.name)
  if (!history.iterations.length) {
    warn('No active learning loop iterations yet.')
    return
  }

  table(
    history.iterations.map((it, i) => [
      i + 1,
      it.timestamp?.slice(0, 19) || 'N/A',
      it.method || it.type || '?',
      it.pool_size || '?',
      it.selected || '?',
      it.added || 0
    ]),
    ['#', 'Timestamp', 'Method', 'Pool', 'Selected', 'Added']
  )
}

// ── Few-shot prompt optimization ─────────────────────

async function runFewShotPrompt(task) {
  header(`Few-shot prompt — "${task.name}" (${task.type})`)

  // Load training data
  const { readJsonl } = await import('./lib/data.js')
  const synPath = join(import.meta.dir, 'data', `${task.name}_synthetic.jsonl`)
  let examples
  try {
    examples = await readJsonl(synPath)
  } catch {
    error('No training data found. Generate data first.')
    return
  }

  if (examples.length < 2) {
    error('Need at least 2 training examples.')
    return
  }

  const strategyIdx = await menu('Selection strategy', [
    'Random', 'Balanced (equal per class)', 'Diverse (max coverage)', 'Similar (to query)'
  ])
  const strategies = ['random', 'balanced', 'diverse', 'similar']
  const strategy = strategies[strategyIdx]

  const kStr = await prompt(`Number of examples (K) [5]:`)
  const k = parseInt(kStr) || 5

  let selected
  if (strategy === 'similar') {
    const query = await prompt('Query text:')
    if (!query) { warn('No query provided.'); return }
    selected = selectExamples(strategy, examples, task, { k, query })
  } else {
    selected = selectExamples(strategy, examples, task, { k })
  }

  info(`Selected ${selected.length} examples via "${strategy}" strategy:`)
  print('')
  for (const ex of selected) {
    print(formatExample(ex, task))
    print('')
  }

  const queryStr = await prompt('Test query (enter to skip):')
  if (queryStr) {
    const fullPrompt = buildFewShotPrompt(task, selected, queryStr)
    info('Generated prompt:')
    print('')
    print(fullPrompt)
  }
}

async function runFewShotOptimize(task) {
  header(`Few-shot optimization — "${task.name}" (${task.type})`)

  const { readJsonl } = await import('./lib/data.js')
  const synPath = join(import.meta.dir, 'data', `${task.name}_synthetic.jsonl`)
  let allData
  try {
    allData = await readJsonl(synPath)
  } catch {
    error('No training data found. Generate data first.')
    return
  }

  if (allData.length < 10) {
    error('Need at least 10 examples (will split into train/val).')
    return
  }

  const kStr = await prompt('Number of few-shot examples (K) [5]:')
  const k = parseInt(kStr) || 5

  const valStr = await prompt('Validation examples [5]:')
  const valCount = Math.min(parseInt(valStr) || 5, allData.length - k)

  // Split: use last valCount as validation, rest as training pool
  const shuffled = [...allData].sort(() => Math.random() - 0.5)
  const valExamples = shuffled.slice(0, valCount)
  const trainExamples = shuffled.slice(valCount)

  info(`Training pool: ${trainExamples.length}, validation: ${valExamples.length}`)

  const apiKey = process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY
  if (!apiKey) {
    error('No API key found (ANTHROPIC_API_KEY or OPENAI_API_KEY).')
    return
  }

  let sp = spinner('Optimizing few-shot examples...')
  try {
    const results = await optimizeFewShot(task, trainExamples, valExamples, {
      k,
      strategies: ['random', 'balanced', 'diverse'],
      randomTrials: 3,
      apiKey,
      onStrategy: name => { sp.text = `Testing strategy: ${name}` }
    })
    sp.stop()

    info('Results (sorted by accuracy):')
    table(
      results.map(r => [
        r.strategy,
        `${(r.accuracy * 100).toFixed(1)}%`,
        `${r.correct}/${r.total}`,
        r.errors.length > 0 ? `${r.errors.length} errors` : '—'
      ]),
      ['Strategy', 'Accuracy', 'Correct', 'Errors']
    )

    if (results.length > 0) {
      const best = results[0]
      info(`Best strategy: ${best.strategy} (${(best.accuracy * 100).toFixed(1)}%)`)

      const save = await confirm('Save best few-shot config?')
      if (save) {
        const config = {
          strategy: best.strategy,
          k,
          examples: best.examples,
          accuracy: best.accuracy,
          evaluatedAt: new Date().toISOString()
        }
        const path = await saveFewShotConfig(task.name, config)
        success(`Saved to ${path}`)
      }
    }
  } catch (e) {
    sp.stop()
    error(`Optimization failed: ${e.message}`)
  }
}

async function runFewShotConfig(task) {
  header(`Few-shot config — "${task.name}"`)

  const config = await loadFewShotConfig(task.name)
  if (!config) {
    warn('No few-shot config saved. Run "Few-shot optimize" first.')
    return
  }

  info(`Strategy: ${config.strategy}`)
  info(`K: ${config.k}`)
  info(`Accuracy: ${(config.accuracy * 100).toFixed(1)}%`)
  info(`Evaluated: ${config.evaluatedAt || 'unknown'}`)

  if (config.examples?.length) {
    info(`\nSaved examples (${config.examples.length}):`)
    print('')
    for (const ex of config.examples) {
      print(formatExample(ex, task))
      print('')
    }
  }
}

async function taskMenu(task) {
  while (true) {
    const action = await menu(`Task: ${task.name}`, [
      'Run full pipeline',                          // 0
      'Preview (sample generation)',                 // 1
      'Preview (streaming)',                         // 2
      'Generate synthetic data',                     // 3
      'Generate (streaming)',                        // 4
      'Prepare data (dedupe + merge + split)',        // 5
      'Semantic dedup (embedding-based)',             // 6
      'Augment data',                                // 7
      'Confidence filter',                           // 8
      'LLM-as-judge quality scoring',                // 9
      'Contrastive example generation',              // 10
      'Cross-provider ensemble generation',          // 11
      'Curriculum analysis',                         // 12
      'Train model (TF-IDF)',                        // 13
      'Train model (embeddings)',                    // 14
      'Train ensemble (all algorithms)',             // 15
      'Compare algorithms',                          // 16
      'Hyperparameter search',                       // 17
      'K-fold cross-validation',                     // 18
      'Predict (interactive)',                       // 19
      'Predict (confidence threshold)',              // 20
      'Predict (ensemble)',                          // 21
      'Zero-shot eval (LLM baseline)',               // 22
      'Uncertainty sampling (active learning)',       // 23
      'Active learning history',                     // 24
      'Evaluation report',                           // 25
      'Feature importance',                          // 26
      'Error taxonomy',                              // 27
      'Calibration analysis',                        // 28
      'Experiment history',                          // 29
      'Model versions',                              // 30
      'Bundle for deployment',                       // 31
      'Embedding cache stats',                       // 32
      'Train transformer (fine-tune)',                 // 33
      'Predict (transformer)',                         // 34
      'Compare all models (experiment history)',        // 35
      'Train CRF (sequence labeling)',                  // 36
      'Predict CRF (tag text)',                         // 37
      'Evaluate CRF (entity F1)',                       // 38
      'Train scoring model (regression)',                // 39
      'Predict scoring (score text)',                    // 40
      'Evaluate scoring (MSE/correlation)',              // 41
      'Active learning loop (any task type)',             // 42
      'Active loop history',                              // 43
      'Few-shot prompt (select examples)',                // 44
      'Few-shot optimize (find best set)',                // 45
      'Few-shot config (view/save)',                      // 46
      '← Back'                                         // 47
    ])

    if (action === 0) await runFullPipeline(task)
    else if (action === 1) await runPreview(task)
    else if (action === 2) await runPreview(task, true)
    else if (action === 3) await runGenerate(task)
    else if (action === 4) await runGenerate(task, true)
    else if (action === 5) await runPrepare(task)
    else if (action === 6) await runSemanticDedup(task)
    else if (action === 7) await runAugment(task)
    else if (action === 8) await runConfidenceFilter(task)
    else if (action === 9) await runLLMJudge(task)
    else if (action === 10) await runContrastive(task)
    else if (action === 11) await runEnsembleGenerate(task)
    else if (action === 12) await runCurriculumAnalysis(task)
    else if (action === 13) {
      const splitResult = await runPrepare(task)
      if (splitResult) await runTrain(task, splitResult)
    }
    else if (action === 14) await runEmbedTrain(task)
    else if (action === 15) await runTrainEnsemble(task)
    else if (action === 16) await runCompare(task)
    else if (action === 17) await runHyperparamSearch(task)
    else if (action === 18) await runKFoldCV(task)
    else if (action === 19) await runPredict(task)
    else if (action === 20) await runThresholdPredict(task)
    else if (action === 21) await runEnsemblePredict(task)
    else if (action === 22) await runZeroShot(task)
    else if (action === 23) await runUncertaintySampling(task)
    else if (action === 24) await runActiveLearningHistory(task)
    else if (action === 25) await runReport(task)
    else if (action === 26) await runFeatureImportance(task)
    else if (action === 27) await runErrorTaxonomy(task)
    else if (action === 28) await runCalibration(task)
    else if (action === 29) await runExperimentHistory(task)
    else if (action === 30) await runModelVersions(task)
    else if (action === 31) await runBundle(task)
    else if (action === 32) await runEmbedCacheStats()
    else if (action === 33) await runTransformerTrain(task)
    else if (action === 34) await runTransformerPredict(task)
    else if (action === 35) await runTransformerCompare(task)
    else if (action === 36) await runCRFTrain(task)
    else if (action === 37) await runCRFPredict(task)
    else if (action === 38) await runCRFEval(task)
    else if (action === 39) await runScoringTrain(task)
    else if (action === 40) await runScoringPredict(task)
    else if (action === 41) await runScoringEval(task)
    else if (action === 42) await runActiveLoop(task)
    else if (action === 43) await runActiveLoopHistory(task)
    else if (action === 44) await runFewShotPrompt(task)
    else if (action === 45) await runFewShotOptimize(task)
    else if (action === 46) await runFewShotConfig(task)
    else return
  }
}

async function main() {
  clear()
  banner()

  const config = await loadConfig()

  if (!API_KEY) {
    warn('ANTHROPIC_API_KEY not set — generation will be unavailable')
    dim('Set it with: export ANTHROPIC_API_KEY=your-key')
  }

  while (true) {
    const tasks = await listTasks()
    const items = [
      'Create new task',
      'Create from template',
      ...tasks.map(t => `Open task: ${t}`),
      'Quit'
    ]

    const choice = await menu('Main Menu', items)

    if (choice === 0) {
      await createTask()
    } else if (choice === 1) {
      await createFromTemplate()
    } else if (choice === items.length - 1) {
      info('Goodbye!')
      process.stdout.write(SHOW_CURSOR)
      process.exit(0)
    } else {
      const taskName = tasks[choice - 2]
      try {
        const task = await loadTask(taskName)
        await taskMenu(task)
      } catch (e) {
        error(e.message)
      }
    }
  }
}

main()
