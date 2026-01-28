# 🧪 Personal LLM Benchmark

A personal benchmark suite for comparing LLM performance across real-world use cases that matter to me.

## Results Summary (2026-01-27)

| Model | Avg Quality | Avg Instruction | Avg Latency | Total Cost |
|-------|-------------|-----------------|-------------|------------|
| gpt-4o | 4.59/5 | 4.41/5 | 8593ms | $0.0689 |
| **gpt-4o-mini** | 4.59/5 | 4.47/5 | 7411ms | $0.0042 |
| gemini-2.5-flash | 4.29/5 | 4.71/5 | 13042ms | $0.0048 |

**Winner:** gpt-4o-mini — matches gpt-4o quality at 1/16th the cost and faster latency.

## Categories Tested

- **Clawdbot**: Personal agent tasks (morning briefs, vault search, LinkedIn drafts)
- **Coding**: Code generation and debugging
- **General**: General knowledge and reasoning

## Usage

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install openai google-generativeai

# Run benchmark
python benchmark.py
```

## Files

- `benchmark.py` - Main benchmark script with test cases and evaluation
- `results.json` - Raw results data
- `results.md` - Human-readable results report

## Key Insights

1. gpt-4o-mini punches way above its weight class
2. Gemini Flash excels at instruction following but has higher latency
3. For personal agent use cases, cheaper models are often "good enough"
