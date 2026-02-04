#!/usr/bin/env python3
"""
Peter's Personal LLM Benchmark
Evaluates models against real workflows: Clawdbot, Coding, General
"""

import os
import json
import time
import random
from dotenv import load_dotenv
import asyncio
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import httpx

# API clients
import anthropic
import openai

# ============================================================================
# Configuration
# ============================================================================

load_dotenv()

MODELS = {
    "claude-sonnet-4.5": {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4.5",
    },
    "gpt-5.2": {
        "provider": "openrouter",
        "model": "openai/gpt-5.2",
    },
    "gpt-5-mini": {
        "provider": "openrouter",
        "model": "openai/gpt-5-mini",
    },
    "gemini-3-pro-preview": {
        "provider": "openrouter",
        "model": "google/gemini-3-pro-preview",
    },
    "gemini-3-flash-preview": {
        "provider": "openrouter",
        "model": "google/gemini-3-flash-preview",
    },
    "kimi-k2.5": {
        "provider": "openrouter",
        "model": "moonshotai/kimi-k2.5",
        "model_env": "KIMI_MODEL",
    },
    "minimax-m2-her": {
        "provider": "openrouter",
        "model": "minimax/minimax-m2-her",
        "model_env": "MINIMAX_MODEL",
    },
}

SUBSCRIPTIONS: dict = {}

# ============================================================================
# Test Prompts
# ============================================================================

TEST_PROMPTS = {
    "clawdbot": [
        {
            "name": "morning_brief",
            "prompt": "Hey, what's on my agenda today? I have a busy schedule with meetings and need to prioritize.",
            "expected": "Structured response with priorities, actionable items",
            "weight": 1.0,
        },
        {
            "name": "vault_search",
            "prompt": "I need to find my notes about bank bonus strategies. What do I have documented?",
            "expected": "Attempts to search/recall, provides structured information",
            "weight": 1.0,
        },
        {
            "name": "linkedin_draft",
            "prompt": "I just finished migrating from a 5,000 line Python AI agent to Clawdbot. Draft a LinkedIn post about this experience. Use my style: conversational, include specific metrics, be honest about trade-offs, short paragraphs, minimal emojis, no em dashes.",
            "expected": "LinkedIn post in Peter's voice, proper structure, specific metrics",
            "weight": 1.5,
        },
        {
            "name": "research_synthesis",
            "prompt": "Research the best bank account bonuses available right now for someone in Florida. Summarize the top 5 options with requirements and payouts.",
            "expected": "Structured research, specific options, actionable info",
            "weight": 1.0,
        },
        {
            "name": "proactive_planning",
            "prompt": "I'm going to be in a conference for the next 3 hours. What are some things you could proactively work on for me based on a typical personal productivity system?",
            "expected": "Shows initiative, suggests relevant tasks, demonstrates judgment",
            "weight": 1.0,
        },
        {
            "name": "task_creation",
            "prompt": "Add 'Book Valentine's Day cabin - URGENT' to my task list. It's for February 14th and I need to do it this week.",
            "expected": "Acknowledges task, confirms details, offers to execute",
            "weight": 0.8,
        },
    ],
    "coding": [
        {
            "name": "simple_function",
            "prompt": "Write a Python function that takes a list of transactions (each with 'amount' and 'category' keys) and returns a dictionary with total spent per category.",
            "expected": "Clean Python, type hints, handles edge cases, docstring",
            "weight": 1.0,
        },
        {
            "name": "debug_code",
            "prompt": """This function sometimes returns None. Find and fix the bug:

def get_user_balance(users: list, user_id: str) -> float:
    for user in users:
        if user['id'] == user_id:
            return user['balance']

# Expected: should return 0.0 if user not found""",
            "expected": "Identifies missing return, explains issue, provides fix",
            "weight": 1.0,
        },
        {
            "name": "code_review",
            "prompt": """Review this code and suggest improvements:

def process(data):
    result = []
    for i in range(len(data)):
        if data[i]['status'] == 'active':
            result.append(data[i]['name'].upper())
    return result""",
            "expected": "Suggests list comprehension, enumerate, type hints, naming",
            "weight": 1.0,
        },
        {
            "name": "api_integration",
            "prompt": "Write a Python async function that fetches data from an API endpoint, handles rate limiting with exponential backoff, and returns the parsed JSON. Include proper error handling.",
            "expected": "Async code, proper error handling, backoff logic, clean structure",
            "weight": 1.2,
        },
        {
            "name": "cli_tool",
            "prompt": "Create a simple Python CLI tool using argparse that takes a directory path and file extension as arguments, then lists all matching files with their sizes.",
            "expected": "Working CLI, proper argparse usage, handles errors",
            "weight": 1.0,
        },
    ],
    "general": [
        {
            "name": "brainstorm",
            "prompt": "I'm an SRE with crypto experience since 2018. Give me 10 side project ideas that combine my SRE skills with crypto/web3. Mix practical and ambitious ideas.",
            "expected": "Creative, relevant to skills, mix of difficulty levels",
            "weight": 1.0,
        },
        {
            "name": "strategy_session",
            "prompt": "Help me think through my Q1 priorities. I want to: pay off debt ($6K remaining), maintain gym consistency (3x/week), post on LinkedIn weekly, and excel at my new SRE job. How should I prioritize?",
            "expected": "Structured advice, trade-off analysis, actionable",
            "weight": 1.0,
        },
        {
            "name": "explain_concept",
            "prompt": "Explain how vector embeddings work for semantic search, like I'm a senior engineer who hasn't worked with ML before.",
            "expected": "Clear explanation, appropriate depth, practical examples",
            "weight": 1.0,
        },
        {
            "name": "compare_options",
            "prompt": "I'm deciding between using Anthropic's Claude API directly ($15/$75 per 1M tokens) vs the $100/month Max subscription. I estimate I make about 20-30 queries per day. Which is better value?",
            "expected": "Does the math, provides recommendation, considers factors",
            "weight": 1.2,
        },
        {
            "name": "pushback_test",
            "prompt": "I'm thinking about quitting my job next week to go full-time on my AI agent side project. It's making $0 right now but I believe in it. Good idea?",
            "expected": "Pushes back thoughtfully, asks questions, doesn't just agree",
            "weight": 1.5,
        },
        {
            "name": "vibe_check",
            "prompt": "Just landed in Vegas for a tech conference. Hotel is nice but I'm exhausted. What should I do tonight?",
            "expected": "Friendly, practical suggestions, acknowledges tiredness",
            "weight": 0.8,
        },
    ],
}

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BenchmarkResult:
    model: str
    category: str
    test_name: str
    prompt: str
    response: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    quality_score: Optional[float] = None
    instruction_score: Optional[float] = None
    error: Optional[str] = None

# ============================================================================
# API Clients
# ============================================================================

def get_anthropic_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)

def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=api_key)

def get_openrouter_client():
    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_COMPAT_API_KEY")
        or os.environ.get("KIMI_API_KEY")
    )
    base_url = (
        os.environ.get("OPENROUTER_BASE_URL")
        or os.environ.get("OPENAI_COMPAT_BASE_URL")
        or os.environ.get("KIMI_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY (or OPENAI_COMPAT_API_KEY/KIMI_API_KEY) not set")
    if not base_url:
        raise ValueError("OPENROUTER_BASE_URL (or OPENAI_COMPAT_BASE_URL/KIMI_BASE_URL) not set")
    headers = {}
    referer = os.environ.get("OPENROUTER_APP_URL") or os.environ.get("OPENROUTER_HTTP_REFERER")
    title = os.environ.get("OPENROUTER_APP_NAME") or os.environ.get("OPENROUTER_X_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    if headers:
        return openai.OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)
    return openai.OpenAI(api_key=api_key, base_url=base_url)

def get_google_client():
    import google.generativeai as genai
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    return genai

# ============================================================================
# Model Runners
# ============================================================================

def run_anthropic(client, model: str, prompt: str) -> tuple[str, float, int, int]:
    """Run prompt through Anthropic API. Returns (response, latency_ms, input_tokens, output_tokens)"""
    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = (time.time() - start) * 1000
    
    text = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    
    return text, latency, input_tokens, output_tokens

def run_openai(client, model: str, prompt: str) -> tuple[str, float, int, int]:
    """Run prompt through OpenAI API."""
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = (time.time() - start) * 1000
    
    text = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    
    return text, latency, input_tokens, output_tokens

def _is_retryable_gemini_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "rate limit" in message
        or "resource_exhausted" in message
        or "quota" in message
        or "429" in message
        or "too many requests" in message
        or "unavailable" in message
        or "503" in message
        or "deadline_exceeded" in message
        or "504" in message
        or "internal" in message
        or "500" in message
    )

def _backoff_delay(attempt: int, initial: float, maximum: float) -> float:
    delay = min(maximum, initial * (2 ** attempt))
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter

def run_google(model: str, prompt: str) -> tuple[str, float, int, int]:
    """Run prompt through Google Gemini API."""
    import tiktoken
    max_retries = int(os.environ.get("GEMINI_MAX_RETRIES", "4"))
    initial_backoff = float(os.environ.get("GEMINI_INITIAL_BACKOFF", "2.0"))
    max_backoff = float(os.environ.get("GEMINI_MAX_BACKOFF", "20.0"))
    
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            start = time.time()
            import google.generativeai as genai
            model_obj = genai.GenerativeModel(model)
            response = model_obj.generate_content(prompt)
            latency = (time.time() - start) * 1000
            
            text = response.text
            # Gemini doesn't always return token counts, estimate
            enc = tiktoken.get_encoding("cl100k_base")
            input_tokens = len(enc.encode(prompt))
            output_tokens = len(enc.encode(text))
            
            return text, latency, input_tokens, output_tokens
        except Exception as e:
            last_error = e
            if not _is_retryable_gemini_error(e) or attempt >= max_retries:
                break
            delay = _backoff_delay(attempt, initial_backoff, max_backoff)
            print(f"  ⏳ Gemini rate limit, retrying in {delay:.1f}s...")
            time.sleep(delay)
    
    if last_error:
        raise last_error
    raise RuntimeError("Gemini request failed without error")

# ============================================================================
# LLM-as-Judge
# ============================================================================

def _parse_judge_json(text: str) -> dict:
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

def judge_response_openai(client, prompt: str, expected: str, response: str, model: str = "gpt-4o") -> tuple[float, float]:
    """Use OpenAI model to judge response quality. Returns (quality_score, instruction_score)"""
    judge_prompt = f"""You are evaluating an LLM response for a personal benchmark.

**Original Prompt:** {prompt}

**Expected Behavior:** {expected}

**Model Response:** {response}

Rate the response on a scale of 1-5 for each criterion:

1. **Quality** (1-5): Is this a high-quality, useful response?
2. **Instruction Following** (1-5): Did it do what was asked?

    Respond in JSON format only:
{{"quality": <1-5>, "instruction": <1-5>, "notes": "<brief explanation>"}}"""

    try:
        result = client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": judge_prompt}]
        )
        
        # Parse JSON from response
        text = result.choices[0].message.content
        data = _parse_judge_json(text)
        return float(data["quality"]), float(data["instruction"])
    except Exception as e:
        print(f"  ⚠️ Judge error: {e}")
        return 3.0, 3.0  # Default middle score

def judge_response_anthropic(client, prompt: str, expected: str, response: str, model: str = "claude-sonnet-4-20250514") -> tuple[float, float]:
    """Use Anthropic model to judge response quality. Returns (quality_score, instruction_score)"""
    judge_prompt = f"""You are evaluating an LLM response for a personal benchmark.

**Original Prompt:** {prompt}

**Expected Behavior:** {expected}

**Model Response:** {response}

Rate the response on a scale of 1-5 for each criterion:

1. **Quality** (1-5): Is this a high-quality, useful response?
2. **Instruction Following** (1-5): Did it do what was asked?

Respond in JSON format only:
{{"quality": <1-5>, "instruction": <1-5>, "notes": "<brief explanation>"}}"""

    try:
        result = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": judge_prompt}]
        )
        
        text = result.content[0].text
        data = _parse_judge_json(text)
        return float(data["quality"]), float(data["instruction"])
    except Exception as e:
        print(f"  ⚠️ Judge error: {e}")
        return 3.0, 3.0  # Default middle score

# ============================================================================
# Model Helpers
# ============================================================================

def _resolve_model_name(config: dict) -> str:
    env_key = config.get("model_env")
    if env_key:
        return os.environ.get(env_key, config["model"])
    return config["model"]

# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_benchmark(models_to_test: list[str] = None, categories: list[str] = None):
    """Run the full benchmark suite."""
    
    if models_to_test is None:
        models_to_test = [
            "claude-sonnet-4.5",
            "gpt-5.2",
            "gpt-5-mini",
            "gemini-3-flash-preview",
            "kimi-k2.5",
            "minimax-m2-her",
        ]
    
    if categories is None:
        categories = ["clawdbot", "coding", "general"]
    
    print("=" * 60)
    print("🧪 Peter's Personal LLM Benchmark")
    print("=" * 60)
    print(f"Models: {', '.join(models_to_test)}")
    print(f"Categories: {', '.join(categories)}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    judge_provider = os.environ.get("JUDGE_PROVIDER", "anthropic").lower()
    judge_model = os.environ.get("JUDGE_MODEL")
    providers_needed = {
        MODELS[m]["provider"] for m in models_to_test if m in MODELS
    }
    if judge_provider in {"anthropic", "openai"}:
        providers_needed.add(judge_provider)

    # Initialize clients only when needed
    clients = {}
    if "anthropic" in providers_needed:
        try:
            clients["anthropic"] = get_anthropic_client()
            print("✅ Anthropic client ready")
        except Exception as e:
            print(f"❌ Anthropic: {e}")
    if "openai" in providers_needed:
        try:
            clients["openai"] = get_openai_client()
            print("✅ OpenAI client ready")
        except Exception as e:
            print(f"❌ OpenAI: {e}")
    if "openrouter" in providers_needed:
        try:
            clients["openrouter"] = get_openrouter_client()
            print("✅ OpenRouter client ready")
        except Exception as e:
            print(f"❌ OpenRouter: {e}")
    if "google" in providers_needed:
        try:
            clients["google"] = get_google_client()
            print("✅ Google client ready")
        except Exception as e:
            print(f"❌ Google: {e}")
    
    print()
    
    results: list[BenchmarkResult] = []
    
    for model_key in models_to_test:
        if model_key not in MODELS:
            print(f"⚠️ Unknown model: {model_key}")
            continue
        
        config = MODELS[model_key]
        provider = config["provider"]
        model_name = _resolve_model_name(config)
        
        if provider not in clients:
            print(f"⚠️ Skipping {model_key} (no {provider} client)")
            continue
        
        print(f"\n🤖 Testing: {model_key}")
        print("-" * 40)
        
        for category in categories:
            if category not in TEST_PROMPTS:
                continue
            
            for test in TEST_PROMPTS[category]:
                test_name = test["name"]
                prompt = test["prompt"]
                expected = test["expected"]
                
                print(f"  📝 {category}/{test_name}...", end=" ", flush=True)
                
                try:
                    # Run the model
                    if provider == "anthropic":
                        response, latency, in_tok, out_tok = run_anthropic(
                            clients["anthropic"], model_name, prompt
                        )
                    elif provider == "openai":
                        response, latency, in_tok, out_tok = run_openai(
                            clients["openai"], model_name, prompt
                        )
                    elif provider == "openrouter":
                        response, latency, in_tok, out_tok = run_openai(
                            clients["openrouter"], model_name, prompt
                        )
                    elif provider == "google":
                        response, latency, in_tok, out_tok = run_google(
                            model_name, prompt
                        )
                    else:
                        raise ValueError(f"Unknown provider: {provider}")
                    
                    # Judge quality (using configured judge)
                    if judge_provider == "anthropic" and "anthropic" in clients:
                        quality, instruction = judge_response_anthropic(
                            clients["anthropic"],
                            prompt,
                            expected,
                            response,
                            model=judge_model or "claude-sonnet-4-20250514",
                        )
                    elif judge_provider == "openai" and "openai" in clients:
                        quality, instruction = judge_response_openai(
                            clients["openai"],
                            prompt,
                            expected,
                            response,
                            model=judge_model or "gpt-4o",
                        )
                    else:
                        quality, instruction = 3.0, 3.0
                    
                    result = BenchmarkResult(
                        model=model_key,
                        category=category,
                        test_name=test_name,
                        prompt=prompt,
                        response=response[:500] + "..." if len(response) > 500 else response,
                        latency_ms=latency,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        cost_usd=0.0,
                        quality_score=quality,
                        instruction_score=instruction,
                    )
                    
                    print(f"✅ Q:{quality:.1f} I:{instruction:.1f} {latency:.0f}ms")
                    
                except Exception as e:
                    result = BenchmarkResult(
                        model=model_key,
                        category=category,
                        test_name=test_name,
                        prompt=prompt,
                        response="",
                        latency_ms=0,
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0,
                        error=str(e),
                    )
                    print(f"❌ {e}")
                
                results.append(result)
    
    return results

# ============================================================================
# Report Generation
# ============================================================================

def generate_report(results: list[BenchmarkResult]) -> str:
    """Generate a simple HTML report from results (scores only)."""
    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # Group by model
    models: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        models.setdefault(r.model, []).append(r)

    # Summary scores
    model_scores: dict[str, dict[str, float]] = {}
    for model, model_results in models.items():
        valid = [r for r in model_results if r.error is None]
        if not valid:
            continue
        avg_quality = avg([r.quality_score or 0 for r in valid])
        avg_instruction = avg([r.instruction_score or 0 for r in valid])
        score = avg([avg_quality, avg_instruction])
        model_scores[model] = {
            "quality": avg_quality,
            "instruction": avg_instruction,
            "score": score,
        }

    # Sort by overall score desc
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]["score"], reverse=True)

    def score_class(value: float) -> str:
        if value >= 4.5:
            return "score score-high"
        if value >= 3.5:
            return "score score-mid"
        return "score score-low"

    def row_html(cols: list[str]) -> str:
        return "<tr>" + "".join(f"<td>{c}</td>" for c in cols) + "</tr>"

    summary_rows = []
    for model, scores in sorted_models:
        summary_rows.append(
            row_html([
                model,
                f"<span class='{score_class(scores['score'])}'>{scores['score']:.2f}</span>",
                f"{scores['quality']:.2f}",
                f"{scores['instruction']:.2f}",
            ])
        )

    category_sections = []
    for category in ["clawdbot", "coding", "general"]:
        rows = []
        for model, model_results in models.items():
            cat_results = [r for r in model_results if r.category == category and r.error is None]
            if not cat_results:
                continue
            avg_q = avg([r.quality_score or 0 for r in cat_results])
            avg_i = avg([r.instruction_score or 0 for r in cat_results])
            score = avg([avg_q, avg_i])
            rows.append(
                row_html([
                    model,
                    f"<span class='{score_class(score)}'>{score:.2f}</span>",
                    f"{avg_q:.2f}",
                    f"{avg_i:.2f}",
                ])
            )
        section = f"""
        <section class="card">
          <h2>{category.title()}</h2>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>Model</th><th>Score</th><th>Quality</th><th>Instruction</th></tr>
              </thead>
              <tbody>
                {''.join(rows) if rows else '<tr><td colspan="4">No results</td></tr>'}
              </tbody>
            </table>
          </div>
        </section>
        """
        category_sections.append(section)

    date_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    tests_run = len(results)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Personal LLM Benchmark</title>
  <style>
    :root {{
      --bg: #0b0f16;
      --card: #121826;
      --text: #e8edf6;
      --muted: #9aa6b2;
      --accent: #7dd3fc;
      --border: #1f2937;
      --high: #22c55e;
      --mid: #f59e0b;
      --low: #ef4444;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Helvetica, Arial, sans-serif;
      background: radial-gradient(1200px 600px at 10% -10%, #1b2335, transparent), var(--bg);
      color: var(--text);
    }}
    .container {{
      max-width: 1100px;
      margin: 40px auto 64px;
      padding: 0 20px;
    }}
    header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 24px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      letter-spacing: 0.2px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 14px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px 18px 12px;
      margin-bottom: 18px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }}
    h2 {{
      margin: 0 0 12px 0;
      font-size: 18px;
      color: var(--accent);
      letter-spacing: 0.3px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--border);
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.6px;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .score {{
      display: inline-block;
      min-width: 62px;
      text-align: center;
      padding: 4px 10px;
      border-radius: 999px;
      font-weight: 600;
      font-size: 12px;
      color: #0b0f16;
    }}
    .score-high {{ background: var(--high); }}
    .score-mid {{ background: var(--mid); }}
    .score-low {{ background: var(--low); }}
    footer {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 18px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>🧪 Personal LLM Benchmark</h1>
      <div class="meta">Run: {date_str} · Tests: {tests_run}</div>
    </header>

    <section class="card">
      <h2>Summary</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Score</th>
              <th>Quality</th>
              <th>Instruction</th>
            </tr>
          </thead>
          <tbody>
            {''.join(summary_rows) if summary_rows else '<tr><td colspan="4">No results</td></tr>'}
          </tbody>
        </table>
      </div>
    </section>

    {''.join(category_sections)}

    <footer>Score = average of Quality and Instruction.</footer>
  </div>
</body>
</html>
"""
    return html

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Use default model list unless arguments are provided
    models = sys.argv[1:] if len(sys.argv) > 1 else None
    
    print("🚀 Starting benchmark...")
    results = run_benchmark(models_to_test=models)
    
    print("\n" + "=" * 60)
    print("📊 Generating report...")
    
    report = generate_report(results)
    
    # Save report
    report_path = os.path.expanduser("~/Documents/Code/personal-llm-benchmark/results.html")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"✅ Report saved to: {report_path}")
    
    # Also save raw JSON
    json_path = os.path.expanduser("~/Documents/Code/personal-llm-benchmark/results.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"✅ Raw data saved to: {json_path}")
    
    # Print a short note
    print("\n" + "=" * 60)
    print("Open results.html to view the benchmark summary.")
