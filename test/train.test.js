import { test, expect, describe, beforeAll, afterAll } from 'bun:test'
import { runTraining } from '../lib/train.js'
import { writeJsonl } from '../lib/data.js'
import { join } from 'node:path'
import { rm, mkdir } from 'node:fs/promises'

const DATA_DIR = join(import.meta.dir, '..', 'data')
const SCRIPTS_DIR = join(import.meta.dir, '..', 'scripts')
const MODELS_DIR = join(import.meta.dir, '..', 'models')

const trainPath = join(DATA_DIR, '_test_train_train.jsonl')
const valPath = join(DATA_DIR, '_test_train_val.jsonl')

beforeAll(async () => {
  await mkdir(DATA_DIR, { recursive: true })

  // Create enough training data for sklearn
  const labels = ['pos', 'neg', 'neutral']
  const templates = {
    pos: ['Great product love it', 'Amazing quality excellent', 'Best thing ever bought', 'Wonderful experience great', 'Highly recommend perfect'],
    neg: ['Terrible broke immediately', 'Worst purchase ever made', 'Complete waste of money', 'Awful quality horrible', 'Very disappointed terrible'],
    neutral: ['Its okay nothing special', 'Average product works fine', 'Decent for the price', 'Not bad not great either', 'Standard quality acceptable']
  }
  const train = []
  const val = []

  for (const label of labels) {
    for (const text of templates[label]) {
      train.push({ text, label })
    }
    val.push({ text: templates[label][0] + ' really', label })
  }

  await writeJsonl(trainPath, train)
  await writeJsonl(valPath, val)
})

afterAll(async () => {
  await rm(trainPath, { force: true })
  await rm(valPath, { force: true })
  await rm(join(MODELS_DIR, '_test_train'), { recursive: true, force: true })
  await rm(join(MODELS_DIR, '_test_train_custom'), { recursive: true, force: true })
  await rm(join(SCRIPTS_DIR, '_test_mock_train.py'), { force: true })
})

describe('runTraining', () => {
  test('trains a classification model successfully', async () => {
    const task = {
      name: '_test_train',
      type: 'classification',
      labels: ['pos', 'neg', 'neutral'],
      training: { script: 'scripts/train.py' }
    }

    const stdoutChunks = []
    const result = await runTraining(task, trainPath, valPath, {
      onStdout: text => stdoutChunks.push(text),
      onStderr: () => {}
    })

    expect(result.modelDir).toContain('_test_train')

    // Check model files were created
    const modelFile = Bun.file(join(result.modelDir, 'model.pkl'))
    expect(await modelFile.exists()).toBe(true)

    const metaFile = Bun.file(join(result.modelDir, 'meta.json'))
    expect(await metaFile.exists()).toBe(true)

    const meta = await metaFile.json()
    expect(meta.task_type).toBe('classification')
    expect(meta.labels).toEqual(['pos', 'neg', 'neutral'])
    expect(meta.accuracy).toBeGreaterThan(0)
    expect(meta.train_size).toBe(15)
    expect(meta.val_size).toBe(3)

    // Verify stdout captured training output
    const fullStdout = stdoutChunks.join('')
    expect(fullStdout).toContain('Training samples: 15')
    expect(fullStdout).toContain('Validation Accuracy')
  }, 30000)

  test('streams stdout to callback', async () => {
    const task = {
      name: '_test_train',
      type: 'classification',
      labels: ['pos', 'neg', 'neutral'],
      training: { script: 'scripts/train.py' }
    }

    const lines = []
    await runTraining(task, trainPath, valPath, {
      onStdout: text => lines.push(text),
      onStderr: () => {}
    })

    expect(lines.length).toBeGreaterThan(0)
  }, 30000)

  test('fails with nonexistent script', async () => {
    const task = {
      name: '_test_train',
      type: 'classification',
      labels: ['a', 'b'],
      training: { script: 'scripts/nonexistent.py' }
    }

    try {
      await runTraining(task, trainPath, valPath, {
        onStdout: () => {},
        onStderr: () => {}
      })
      expect(true).toBe(false) // should not reach
    } catch (e) {
      expect(e.message).toBeTruthy()
    }
  }, 10000)

  test('passes extra args to training script', async () => {
    // Create a mock script that just prints its args
    const mockScript = join(SCRIPTS_DIR, '_test_mock_train.py')
    await Bun.write(mockScript, `
import sys
print('ARGS:', ' '.join(sys.argv[1:]))
`)

    const task = {
      name: '_test_train_custom',
      type: 'classification',
      labels: ['a', 'b'],
      training: {
        script: 'scripts/_test_mock_train.py',
        args: { epochs: 5, 'learning-rate': 0.001 }
      }
    }

    const stdoutChunks = []
    await runTraining(task, trainPath, valPath, {
      onStdout: text => stdoutChunks.push(text),
      onStderr: () => {}
    })

    const stdout = stdoutChunks.join('')
    expect(stdout).toContain('--epochs')
    expect(stdout).toContain('5')
    expect(stdout).toContain('--learning-rate')
    expect(stdout).toContain('0.001')
  }, 10000)
})
