# Synthetic Data Generation

The generation module (`lib/generate.js`) calls the Anthropic Messages API directly using `fetch` — no SDK or dependencies. It produces batches of labeled examples and writes them to a JSONL file.

## How it works

1. A **system prompt** is built from the task definition, telling Claude it's a training data generator and describing the task type, labels, or fields.
2. A **user prompt** is built for each batch, requesting a specific number of examples as a JSON array.
3. The response is parsed by extracting the first JSON array from Claude's output (handles markdown fences gracefully).
4. Batches accumulate until the target count is reached.
5. Results are written to `data/<task-name>_synthetic.jsonl`.

## API call structure

Each batch request hits `https://api.anthropic.com/v1/messages` with:

- **model** — from `task.synthetic.model` (default: `claude-sonnet-4-20250514`)
- **max_tokens** — 4096
- **system** — task-aware system prompt
- **messages** — single user message requesting the batch

The API key is read from the `ANTHROPIC_API_KEY` environment variable.

## System prompt construction

The system prompt varies by task type:

**Classification:**
```
You are a training data generator. Your job is to produce realistic, diverse examples for a machine learning task.

Task: Classify customer review sentiment
Type: classification
Labels: positive, negative, neutral
```

**Extraction:**
```
You are a training data generator. Your job is to produce realistic, diverse examples for a machine learning task.

Task: Extract contact information from emails
Type: extraction
Fields to extract: name, email, phone
```

## Batch prompt

Each batch prompt instructs Claude to return a JSON array of the exact batch size, with the format appropriate for the task type:

- **Classification:** `{"text": "...", "label": "..."}`
- **Extraction:** `{"text": "...", "fields": {...}}`
- **Regression:** `{"text": "...", "value": ...}`

For classification tasks, the prompt also asks Claude to distribute labels roughly evenly.

## Batching

Generation happens in sequential batches to keep individual API responses manageable. With `count: 200` and `batchSize: 10`, the generator makes 20 API calls producing 10 examples each. The TUI shows a progress bar as batches complete.

Smaller batch sizes (5–10) tend to produce higher quality examples. Larger batches (20+) are faster but may produce more repetitive data.

## Output format

The output is JSONL (one JSON object per line):

```jsonl
{"text": "Absolutely love this product! Best purchase I've made.", "label": "positive"}
{"text": "Broke after two days. Complete waste of money.", "label": "negative"}
{"text": "It works as described. Nothing remarkable.", "label": "neutral"}
```

## Cost considerations

Each batch is a single API call. With the default model (`claude-sonnet-4-20250514`), rough costs depend on example complexity. For typical classification tasks with short text examples, generating 500 examples in batches of 10 costs a few cents.

To control costs, adjust `synthetic.count` and `synthetic.batchSize` in your task definition.

## Error handling

- HTTP errors from the API (rate limits, auth failures) throw with the status code and response body.
- If Claude's response doesn't contain a valid JSON array, the batch fails with a parse error.
- The generator does not retry failed batches — it throws immediately. You can re-run generation and it will overwrite the previous output.
