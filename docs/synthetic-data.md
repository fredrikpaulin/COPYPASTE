# Synthetic Data Generation

The generation module (`lib/generate.js`) calls LLM APIs directly using `fetch` — no SDK or dependencies. It supports multiple providers (Anthropic Claude, OpenAI, Ollama) via the provider abstraction in `lib/provider.js`. It produces batches of labeled examples and writes them to a JSONL file.

## How it works

1. A **system prompt** is built from the task definition, telling Claude it's a training data generator and describing the task type, labels, or fields.
2. A **user prompt** is built for each batch, requesting a specific number of examples as a JSON array.
3. The response is parsed by extracting the first JSON array from Claude's output (handles markdown fences gracefully).
4. Batches accumulate until the target count is reached.
5. Results are written to `data/<task-name>_synthetic.jsonl`.

## Providers

Generation supports three LLM providers. Set `synthetic.provider` in your task definition to choose one:

| Provider | Key | Env variable | Default model | Endpoint |
|---|---|---|---|---|
| Anthropic (Claude) | `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` | `https://api.anthropic.com/v1/messages` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` | `https://api.openai.com/v1/chat/completions` |
| Ollama (local) | `ollama` | _(none)_ | `llama3` | `http://localhost:11434/api/chat` |

If no provider is specified, `anthropic` is used by default. You can also set the provider globally in `distill.config.json`.

## API call structure

Each batch request hits the provider's API endpoint with:

- **model** — from `task.synthetic.model` (or the provider's default)
- **max_tokens** — 4096
- **system** — task-aware system prompt
- **messages** — single user message requesting the batch

The API key is read from the provider's environment variable (see table above). Ollama requires no API key.

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

## Retries and backoff

API calls use exponential backoff with jitter. If Claude returns a 429 (rate limited), 529 (overloaded), or any 5xx status, the generator retries up to `maxRetries` times (default 3, configurable via `distill.config.json`). If the response includes a `retry-after` header, that value is used instead of the computed backoff.

The backoff formula is `base * 2^attempt`, capped at a maximum of 60 seconds, then jittered to a random value in the lower half of that range. Client errors (4xx other than 429) fail immediately without retry.

## Input validation

Every generated example is validated against the task definition before it's written to disk:

- **All types** — must have a non-empty `text` string
- **Classification** — `label` must be a string present in the task's `labels` array
- **Extraction** — must have a `fields` object
- **Regression** — `value` must be a number

Malformed examples are silently dropped. The TUI reports how many were dropped at the end of each generation run.

## Preview mode

The `preview` command generates a small sample (default 5 examples) without writing to disk. Use it to iterate on your prompt before committing to a full generation run. In the TUI, select **Preview data** from the task menu.

## Error handling

- HTTP errors trigger retries for 429, 529, and 5xx. Other status codes throw immediately.
- If Claude's response doesn't contain a valid JSON array, the batch fails with a parse error.
- Individual invalid examples within a valid batch are dropped — the rest of the batch is kept.
- Generation progress is reported per batch via the TUI progress bar.
