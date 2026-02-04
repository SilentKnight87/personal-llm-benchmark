# 🧪 Personal LLM Benchmark

A personal benchmark suite for comparing LLM performance across real-world use cases that matter to me.

## Categories Tested

- **Clawdbot**: Personal agent tasks (morning briefs, vault search, LinkedIn drafts)
- **Coding**: Code generation and debugging
- **General**: General knowledge and reasoning

## Usage

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install openai anthropic python-dotenv

# Run benchmark
python benchmark.py
```
Open `results.html` in your browser to view scores.

## Configuration Notes

- **API keys** (set in `.env`):
  - `ANTHROPIC_API_KEY` (judge)
  - `OPENROUTER_API_KEY` (all benchmark models)
- **Judge model**: defaults to Anthropic `claude-sonnet-4-20250514`
  - Override with `JUDGE_PROVIDER` (`anthropic` or `openai`) and `JUDGE_MODEL`
  - If you choose `openai`, set `OPENAI_API_KEY`
- **OpenRouter**:
  - `OPENROUTER_API_KEY` (required)
  - `OPENROUTER_BASE_URL` (optional, defaults to `https://openrouter.ai/api/v1`)
  - Optional attribution headers: `OPENROUTER_APP_URL` (HTTP-Referer), `OPENROUTER_APP_NAME` (X-Title)
  - Optional overrides: `KIMI_MODEL`, `MINIMAX_MODEL`
- **Gemini rate limits** (only if you use direct Google API): `GEMINI_MAX_RETRIES`, `GEMINI_INITIAL_BACKOFF`, `GEMINI_MAX_BACKOFF`
- **Direct Gemini deps** (optional): `pip install google-generativeai tiktoken`

## `.env` Example

```
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
OPENROUTER_APP_URL=https://your.app
OPENROUTER_APP_NAME=Personal LLM Benchmark
```

## Default Models

By default, the benchmark runs:
`claude-sonnet-4.5`, `gpt-5.2`, `gpt-5-mini`, `gemini-3-flash-preview`, `kimi-k2.5`, `minimax-m2-her`

## Files

- `benchmark.py` - Main benchmark script with test cases and evaluation
- `results.json` - Raw results data
- `results.html` - Human-readable HTML results report

## Notes

- Benchmark results are generated in `results.html` and focus on scores only (quality + instruction).
