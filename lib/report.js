import { join } from 'node:path'
import { mkdir } from 'node:fs/promises'

const REPORTS_DIR = join(import.meta.dir, '..', 'reports')

// Build confusion matrix from predictions
function confusionMatrix(actual, predicted, labels) {
  const matrix = {}
  for (const a of labels) {
    matrix[a] = {}
    for (const p of labels) matrix[a][p] = 0
  }
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i], p = predicted[i]
    if (matrix[a] && matrix[a][p] !== undefined) matrix[a][p]++
  }
  return matrix
}

// Compute per-label precision, recall, F1
function perLabelMetrics(actual, predicted, labels) {
  const metrics = {}
  for (const label of labels) {
    let tp = 0, fp = 0, fn = 0
    for (let i = 0; i < actual.length; i++) {
      if (predicted[i] === label && actual[i] === label) tp++
      else if (predicted[i] === label && actual[i] !== label) fp++
      else if (predicted[i] !== label && actual[i] === label) fn++
    }
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0
    const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0
    metrics[label] = { precision, recall, f1, support: tp + fn }
  }
  return metrics
}

// Find misclassified examples
function findErrors(data, predictions, { maxPerLabel = 3 } = {}) {
  const errors = {}
  for (let i = 0; i < data.length; i++) {
    const actual = data[i].label
    const predicted = predictions[i]?.label || predictions[i]
    if (actual !== predicted) {
      const key = `${actual} → ${predicted}`
      if (!errors[key]) errors[key] = []
      if (errors[key].length < maxPerLabel) {
        errors[key].push({ text: data[i].text, actual, predicted })
      }
    }
  }
  return errors
}

// Generate HTML evaluation report
async function generateReport(taskName, { valData, predictions, labels, meta }) {
  const actual = valData.map(d => d.label)
  const preds = predictions.map(p => p.label || p)
  const cm = confusionMatrix(actual, preds, labels)
  const metrics = perLabelMetrics(actual, preds, labels)
  const errors = findErrors(valData, predictions)
  const accuracy = meta?.accuracy || (actual.filter((a, i) => a === preds[i]).length / actual.length)

  // Dataset composition
  const labelDist = {}
  for (const d of valData) {
    labelDist[d.label] = (labelDist[d.label] || 0) + 1
  }

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation Report: ${taskName}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; padding: 2rem; background: #f8f9fa; color: #1a1a2e; }
  h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
  h2 { font-size: 1.3rem; margin: 2rem 0 1rem; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.5rem; }
  .meta { color: #666; font-size: 0.9rem; margin-bottom: 2rem; }
  .accuracy-badge { display: inline-block; background: #10b981; color: white; padding: 0.3rem 0.8rem; border-radius: 1rem; font-weight: 600; font-size: 1.1rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: center; font-size: 0.9rem; }
  th { background: #f1f5f9; font-weight: 600; }
  .highlight { background: #dbeafe; font-weight: 600; }
  .error-group { margin: 1rem 0; }
  .error-group h3 { font-size: 1rem; color: #dc2626; margin-bottom: 0.5rem; }
  .error-example { background: white; border: 1px solid #e5e7eb; border-radius: 0.5rem; padding: 0.75rem; margin: 0.5rem 0; font-size: 0.85rem; }
  .bar { height: 1.2rem; border-radius: 0.3rem; display: inline-block; }
  .bar-container { background: #e5e7eb; border-radius: 0.3rem; width: 100%; height: 1.2rem; }
  .chart-row { display: flex; align-items: center; gap: 0.5rem; margin: 0.3rem 0; }
  .chart-label { width: 100px; text-align: right; font-size: 0.85rem; }
  .chart-value { font-size: 0.85rem; color: #666; min-width: 40px; }
  footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e0e0e0; color: #999; font-size: 0.8rem; }
</style>
</head>
<body>
<h1>Evaluation Report: ${taskName}</h1>
<p class="meta">
  Generated ${new Date().toISOString().slice(0, 19)} &middot;
  Algorithm: ${meta?.algorithm || 'logistic_regression'} &middot;
  Train: ${meta?.train_size || '?'} &middot; Val: ${meta?.val_size || '?'}
</p>
<p><span class="accuracy-badge">Accuracy: ${(accuracy * 100).toFixed(1)}%</span></p>

<h2>Confusion Matrix</h2>
<table>
  <tr><th>Actual \\ Predicted</th>${labels.map(l => `<th>${l}</th>`).join('')}</tr>
  ${labels.map(a => `<tr><th>${a}</th>${labels.map(p => {
    const val = cm[a][p]
    const cls = a === p ? ' class="highlight"' : ''
    return `<td${cls}>${val}</td>`
  }).join('')}</tr>`).join('\n  ')}
</table>

<h2>Per-Label Metrics</h2>
<table>
  <tr><th>Label</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
  ${labels.map(l => {
    const m = metrics[l]
    return `<tr><td>${l}</td><td>${m.precision.toFixed(3)}</td><td>${m.recall.toFixed(3)}</td><td>${m.f1.toFixed(3)}</td><td>${m.support}</td></tr>`
  }).join('\n  ')}
</table>

<h2>Dataset Composition</h2>
${labels.map(l => {
    const count = labelDist[l] || 0
    const pct = valData.length > 0 ? (count / valData.length * 100) : 0
    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']
    const color = colors[labels.indexOf(l) % colors.length]
    return `<div class="chart-row">
  <span class="chart-label">${l}</span>
  <div class="bar-container"><div class="bar" style="width:${pct}%;background:${color}"></div></div>
  <span class="chart-value">${count} (${pct.toFixed(0)}%)</span>
</div>`
  }).join('\n')}

<h2>Example Errors</h2>
${Object.keys(errors).length === 0 ? '<p>No misclassifications found.</p>' :
  Object.entries(errors).map(([key, examples]) => `
<div class="error-group">
  <h3>${key} (${examples.length} shown)</h3>
  ${examples.map(e => `<div class="error-example">"${e.text}"</div>`).join('\n  ')}
</div>`).join('\n')}

<footer>Generated by copypaste — Feature Distillation Pipeline</footer>
</body>
</html>`

  await mkdir(REPORTS_DIR, { recursive: true })
  const outPath = join(REPORTS_DIR, `${taskName}_report.html`)
  await Bun.write(outPath, html)
  return { path: outPath, accuracy, metrics, confusionMatrix: cm }
}

export { generateReport, confusionMatrix, perLabelMetrics, findErrors }
