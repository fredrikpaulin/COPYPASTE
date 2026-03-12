import { join } from 'node:path'
import { mkdir, cp } from 'node:fs/promises'

const MODELS_DIR = join(import.meta.dir, '..', 'models')

// Package a trained model as a standalone module
async function bundle(taskName, outputDir) {
  const modelDir = join(MODELS_DIR, taskName)
  const metaFile = Bun.file(join(modelDir, 'meta.json'))

  if (!await metaFile.exists()) {
    throw new Error(`No model found for task "${taskName}"`)
  }

  const meta = await metaFile.json()
  await mkdir(outputDir, { recursive: true })

  // Copy model artifacts
  await cp(join(modelDir, 'model.pkl'), join(outputDir, 'model.pkl'))
  await cp(join(modelDir, 'meta.json'), join(outputDir, 'meta.json'))

  // Copy ONNX if it exists
  const onnxFile = Bun.file(join(modelDir, 'model.onnx'))
  if (await onnxFile.exists()) {
    await cp(join(modelDir, 'model.onnx'), join(outputDir, 'model.onnx'))
  }

  // Generate a standalone predict script
  const predictScript = `#!/usr/bin/env python3
"""Standalone prediction script for ${taskName}."""
import json, pickle, sys, os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

def load():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def predict(texts):
    model = load()
    preds = model.predict(texts)
    probas = model.predict_proba(texts)
    return [{'text': t, 'label': p, 'confidence': float(max(pr))}
            for t, p, pr in zip(texts, preds, probas)]

if __name__ == '__main__':
    if len(sys.argv) > 1:
        results = predict(sys.argv[1:])
    else:
        texts = [json.loads(l)['text'] for l in sys.stdin if l.strip()]
        results = predict(texts)
    print(json.dumps(results, indent=2))
`
  await Bun.write(join(outputDir, 'predict.py'), predictScript)

  // Generate a package.json for the bundle
  const pkg = {
    name: `${taskName}-model`,
    version: '1.0.0',
    description: `Distilled ${meta.task_type} model for: ${taskName}`,
    type: 'module',
    main: 'predict.py',
    metadata: {
      task_type: meta.task_type,
      labels: meta.labels,
      accuracy: meta.accuracy,
      created_at: meta.created_at
    }
  }
  await Bun.write(join(outputDir, 'package.json'), JSON.stringify(pkg, null, 2) + '\n')

  // Generate a README
  const readme = `# ${taskName} model

Distilled ${meta.task_type} model.

**Labels:** ${(meta.labels || []).join(', ')}
**Accuracy:** ${meta.accuracy?.toFixed(4) || 'N/A'}
**Training size:** ${meta.train_size || 'N/A'}

## Usage

\`\`\`bash
python3 predict.py "your text here"
\`\`\`

Or via stdin (JSONL):

\`\`\`bash
echo '{"text": "your text here"}' | python3 predict.py
\`\`\`

## Python API

\`\`\`python
from predict import predict
results = predict(["your text here"])
print(results)
\`\`\`
`
  await Bun.write(join(outputDir, 'README.md'), readme)

  return {
    path: outputDir,
    files: ['model.pkl', 'meta.json', 'predict.py', 'package.json', 'README.md']
  }
}

export { bundle }
