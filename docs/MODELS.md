# Using a real model

Out of the box the specialists run on a deterministic stand-in, so everything works offline and costs
nothing. To run them on a real language model, set two environment variables. No code changes.

## Quick start

The recommended setup is [OpenRouter](https://openrouter.ai): one key reaches hosted open-weights
models for fractions of a cent per case.

```bash
cp .env.example .env    # then set CASEFILE_PROVIDER=openrouter and OPENROUTER_API_KEY in it
uv sync
uv run python scripts/estimate_cost.py   # what a run will cost, without calling anything
uv run python scripts/demo.py --fresh    # review the cases at http://127.0.0.1:8000
```

`uv sync` installs both provider SDKs for development. To install only one: `uv sync --extra openai`
for OpenRouter and the other OpenAI-compatible hosts, or `uv sync --extra anthropic` for Claude.

## Which provider

| Provider | Set | Default model | Good for |
|---|---|---|---|
| `openrouter` (recommended) | `OPENROUTER_API_KEY` | `deepseek/deepseek-v4-pro` | the best quality per dollar; one key, many models |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` | the highest quality; Sonnet 5.5 and Haiku 5.5 trade some quality for cost |
| `together`, `groq`, `fireworks`, `deepinfra` | that host's key and `CASEFILE_MODEL` | none, pick one | a host you already use; Groq for speed |
| `ollama` | `CASEFILE_MODEL` | none | running on your own machine (needs enough memory) |
| `openai-compatible` | `CASEFILE_BASE_URL`, `CASEFILE_API_KEY`, `CASEFILE_MODEL` | none | any other server that speaks the OpenAI chat API, such as vLLM |

Quality comes first. Every fact the model extracts has to quote the exact words it read, and a
check outside the model rejects anything the document does not say. So a cheaper model costs
accuracy, never correctness: its mistakes show up as rejected facts and requests for information,
not as wrong approvals. The recommendation itself always comes from fixed rules, never the model.

Any model id your provider lists works with `CASEFILE_MODEL`. Defaults and prices were checked on
2026-10-25 and will drift; override them when they do.

## What it costs

`scripts/estimate_cost.py` runs the real workflow offline, measures the exact prompts each case would
send, and prices them for the configured provider. On 2026-10-25 it estimated, per generated case
(about five short calls):

| Model | Per case | 150 cases |
|---|---|---|
| `deepseek/deepseek-v4-pro` via OpenRouter | about $0.0006 | about $0.09 |
| `claude-opus-5-5` | about $0.016, plus its reasoning tokens | about $2.42, plus reasoning |

These are estimates, at about four characters per token. Models that reason before answering bill
that reasoning as output, which the estimate cannot see. The review screen shows the actual tokens
and cost of every call once a case has run.

## What happens on a call

- **Prompts.** Each document goes to the extractor JSON-encoded, so nothing inside it can close the
  fence. The model is told to ignore any instruction inside it, and it never decides: the rules do.
- **Checks.** A fact counts only if its quote is the whole value of that fact's field in the
  document, and every field the model leaves out is added back and flagged to the reviewer.
- **Budgets.** Call and cost limits are checked before each step. A cost limit with no known price
  stops the run rather than guessing. With Claude, a refused request can be answered by the
  fallback model, which is billed at its own rate while the estimate here uses the configured one.
- **Output.** Claude gets a JSON schema it must follow (structured outputs). OpenAI-compatible hosts
  get JSON mode, with the schema stated in the prompt. A reply that still isn't valid is recorded
  and rejected, never guessed at.
- **Failures.** Both SDKs retry rate limits and server errors. A Claude refusal is retried on
  Anthropic's recommended fallback model (the server-side `fallbacks` option); if that also
  declines, the step fails loudly.
- **Records.** Every answer, with its token counts and cost, is written to the case's event log, so
  resuming after a crash never pays for the same answer twice. Answers are also cached per provider
  and model, so regenerating the demo doesn't either.

## Settings

| Variable | Meaning |
|---|---|
| `CASEFILE_PROVIDER` | one of the providers above; `reference` (the stand-in) when unset |
| `CASEFILE_MODEL` | model id, overriding the provider's default |
| `CASEFILE_BASE_URL` | endpoint, overriding the provider's |
| `CASEFILE_API_KEY` | key, overriding the provider's own variable |
| `CASEFILE_EFFORT` | Claude only: `low`, `medium`, `high`, `xhigh` or `max` |
| `CASEFILE_PRICE_INPUT`, `CASEFILE_PRICE_OUTPUT` | USD per million tokens, for cost figures on any model |

Variables set in your shell win over the `.env` file. Keys are never written into the repository,
the event log or the page.
