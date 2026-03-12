import {
  clear, banner, header, success, warn, error, info, dim,
  progress, spinner, prompt, menu, table, SHOW_CURSOR
} from './lib/tui.js'
import { loadTask, listTasks, saveTask } from './lib/task.js'
import { generate, preview } from './lib/generate.js'
import { loadAndMerge, split, stats, deduplicate, augment, labelImbalance, filterByConfidence } from './lib/data.js'
import { runTraining, versionModel, listVersions } from './lib/train.js'
import { predict, loadMeta, listModels } from './lib/infer.js'
import { bundle } from './lib/bundle.js'
import { getUncertainExamples, generateAndRankByUncertainty, llmLabel, loadHistory, saveIteration } from './lib/active.js'
import { listProviders, resolveProvider } from './lib/provider.js'
import { listTemplates, loadTemplate } from './lib/templates.js'
import { generateReport } from './lib/report.js'
import { loadConfig } from './lib/config.js'
import { startLog, logEntry, flushLog } from './lib/log.js'

const API_KEY = process.env.ANTHROPIC_API_KEY

// Graceful exit
process.on('SIGINT', () => { process.stdout.write(SHOW_CURSOR); process.exit(0) })

async function createTask() {
  header('Create New Task')

  const name = await prompt('Task name (lowercase, hyphens):')
  const typeIdx = await menu('Task type:', ['classification', 'extraction', 'regression', 'sequence-labeling'])
  const type = ['classification', 'extraction', 'regression', 'sequence-labeling'][typeIdx]
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

async function runPreview(task) {
  if (!API_KEY) { error('Set ANTHROPIC_API_KEY in your environment'); return }
  if (!task.synthetic) { warn('No synthetic config on this task'); return }

  header(`Preview for "${task.name}"`)
  const sp = spinner('Generating sample...')

  try {
    const result = await preview(task, {
      apiKey: API_KEY,
      count: 5,
      onRetry: ({ attempt, waitMs }) => {
        sp.stop(`Retry ${attempt} in ${Math.round(waitMs / 1000)}s...`)
      }
    })
    sp.stop(`${result.examples.length} examples generated`)

    for (const ex of result.examples) {
      info(JSON.stringify(ex))
    }
    if (result.dropped) {
      warn(`${result.dropped} malformed examples dropped`)
    }
  } catch (e) {
    sp.fail('Preview failed')
    error(e.message)
  }
}

async function runGenerate(task) {
  if (!API_KEY) { error('Set ANTHROPIC_API_KEY in your environment'); return }
  if (!task.synthetic) { warn('No synthetic config on this task'); return }

  await startLog(task.name)
  logEntry('generate_start', { task: task.name, count: task.synthetic.count })

  header(`Generating synthetic data for "${task.name}"`)
  info(`Model: ${task.synthetic.model || 'claude-sonnet-4-20250514'}`)
  info(`Target: ${task.synthetic.count} examples in batches of ${task.synthetic.batchSize}`)

  try {
    const result = await generate(task, {
      apiKey: API_KEY,
      onProgress: (current, total) => progress(current, total, 'examples'),
      onRetry: ({ attempt, waitMs, status }) => {
        warn(`Rate limited (${status}), retry ${attempt} in ${Math.round(waitMs / 1000)}s...`)
        logEntry('retry', { attempt, waitMs, status })
      },
      onDropped: count => {
        warn(`${count} malformed examples dropped during validation`)
        logEntry('dropped', { count })
      }
    })
    success(`Generated ${result.count} examples → ${result.path}`)
    if (result.dropped) dim(`  (${result.dropped} dropped)`)
    logEntry('generate_complete', { count: result.count, dropped: result.dropped })
  } catch (e) {
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

async function runTrain(task, splitResult, { algorithm, compare = false, search = false, grid } = {}) {
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
  try {
    const result = await runTraining(task, splitResult.train.path, splitResult.val.path, {
      onnx: !compare && !search,
      algorithm,
      compare,
      search,
      grid,
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
    success(`Model saved to ${result.modelDir}`)
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

async function taskMenu(task) {
  while (true) {
    const action = await menu(`Task: ${task.name}`, [
      'Run full pipeline',
      'Preview (sample generation)',
      'Generate synthetic data',
      'Prepare data (dedupe + merge + split)',
      'Augment data',
      'Confidence filter',
      'Train model',
      'Compare algorithms',
      'Hyperparameter search',
      'Predict (interactive)',
      'Uncertainty sampling (active learning)',
      'Active learning history',
      'Evaluation report',
      'Model versions',
      'Bundle for deployment',
      '← Back'
    ])

    if (action === 0) await runFullPipeline(task)
    else if (action === 1) await runPreview(task)
    else if (action === 2) await runGenerate(task)
    else if (action === 3) await runPrepare(task)
    else if (action === 4) await runAugment(task)
    else if (action === 5) await runConfidenceFilter(task)
    else if (action === 6) {
      const splitResult = await runPrepare(task)
      if (splitResult) await runTrain(task, splitResult)
    }
    else if (action === 7) await runCompare(task)
    else if (action === 8) await runHyperparamSearch(task)
    else if (action === 9) await runPredict(task)
    else if (action === 10) await runUncertaintySampling(task)
    else if (action === 11) await runActiveLearningHistory(task)
    else if (action === 12) await runReport(task)
    else if (action === 13) await runModelVersions(task)
    else if (action === 14) await runBundle(task)
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
