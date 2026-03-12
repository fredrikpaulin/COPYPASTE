import {
  clear, banner, header, success, warn, error, info, dim,
  progress, spinner, prompt, menu, table, SHOW_CURSOR
} from './lib/tui.js'
import { loadTask, listTasks, saveTask } from './lib/task.js'
import { generate } from './lib/generate.js'
import { loadAndMerge, split, stats } from './lib/data.js'
import { runTraining } from './lib/train.js'

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

async function runGenerate(task) {
  if (!API_KEY) {
    error('Set ANTHROPIC_API_KEY in your environment')
    return
  }
  if (!task.synthetic) {
    warn('No synthetic config on this task')
    return
  }

  header(`Generating synthetic data for "${task.name}"`)
  info(`Model: ${task.synthetic.model || 'claude-sonnet-4-20250514'}`)
  info(`Target: ${task.synthetic.count} examples in batches of ${task.synthetic.batchSize}`)

  try {
    const result = await generate(task, {
      apiKey: API_KEY,
      onProgress: (current, total) => progress(current, total, `examples`)
    })
    success(`Generated ${result.count} examples → ${result.path}`)
  } catch (e) {
    error(`Generation failed: ${e.message}`)
  }
}

async function runPrepare(task) {
  header(`Preparing data for "${task.name}"`)

  const data = await loadAndMerge(task)
  if (!data.length) {
    warn('No data found. Generate synthetic data first.')
    return null
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

  const result = await split(task, data)
  success(`Train: ${result.train.count} → ${result.train.path}`)
  success(`Val: ${result.val.count} → ${result.val.path}`)
  return result
}

async function runTrain(task, splitResult) {
  header(`Training model for "${task.name}"`)

  const sp = spinner('Training in progress...')
  try {
    const result = await runTraining(task, splitResult.train.path, splitResult.val.path, {
      onStdout: text => {
        for (const line of text.split('\n').filter(Boolean)) {
          if (line.includes('Accuracy')) {
            sp.stop(line.trim())
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

async function runFullPipeline(task) {
  await runGenerate(task)
  const splitResult = await runPrepare(task)
  if (splitResult) {
    await runTrain(task, splitResult)
  }
}

async function taskMenu(task) {
  while (true) {
    const action = await menu(`Task: ${task.name}`, [
      'Run full pipeline',
      'Generate synthetic data',
      'Prepare data (merge + split)',
      'Train model',
      '← Back'
    ])

    if (action === 0) await runFullPipeline(task)
    else if (action === 1) await runGenerate(task)
    else if (action === 2) await runPrepare(task)
    else if (action === 3) {
      const splitResult = await runPrepare(task)
      if (splitResult) await runTrain(task, splitResult)
    }
    else return
  }
}

async function main() {
  clear()
  banner()

  if (!API_KEY) {
    warn('ANTHROPIC_API_KEY not set — generation will be unavailable')
    dim('Set it with: export ANTHROPIC_API_KEY=your-key')
  }

  while (true) {
    const tasks = await listTasks()
    const items = [
      'Create new task',
      ...tasks.map(t => `Open task: ${t}`),
      'Quit'
    ]

    const choice = await menu('Main Menu', items)

    if (choice === 0) {
      await createTask()
    } else if (choice === items.length - 1) {
      info('Goodbye!')
      process.stdout.write(SHOW_CURSOR)
      process.exit(0)
    } else {
      const taskName = tasks[choice - 1]
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
