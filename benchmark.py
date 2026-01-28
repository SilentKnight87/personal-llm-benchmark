#!/usr/bin/env python3
"""
Peter's Personal LLM Benchmark
Evaluates models against real workflows: Clawdbot, Coding, General
"""

import os
import json
import time
import asyncio
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import httpx

# API clients
import anthropic
import openai
import google.generativeai as genai

# Token counting
import tiktoken

# ============================================================================
# Configuration
# ============================================================================

MODELS = {
    "claude-opus": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",  # Using Sonnet for benchmark (Opus via subscription)
        "input_price": 3.0,   # per 1M tokens
        "output_price": 15.0,
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "input_price": 3.0,
        "output_price": 15.0,
    },
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "input_price": 2.5,
        "output_price": 10.0,
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "input_price": 0.15,
        "output_price": 0.6,
    },
    "gemini-2.5-pro": {
        "provider": "google",
        "model": "gemini-2.5-pro",
        "input_price": 1.25,
        "output_price": 5.0,
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "model": "gemini-2.5-flash",
        "input_price": 0.15,
        "output_price": 0.6,
    },
}

SUBSCRIPTIONS = {
    "claude_max": {"cost": 100, "models": ["claude-opus", "claude-sonnet"]},
    "chatgpt_plus": {"cost": 20, "models": ["gpt-4o", "gpt-4o-mini"]},
    "gemini_pro": {"cost": 20, "models": ["gemini-2.5-pro"]},
}

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

def get_google_client():
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

def run_google(model: str, prompt: str) -> tuple[str, float, int, int]:
    """Run prompt through Google Gemini API."""
    start = time.time()
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(prompt)
    latency = (time.time() - start) * 1000
    
    text = response.text
    # Gemini doesn't always return token counts, estimate
    enc = tiktoken.get_encoding("cl100k_base")
    input_tokens = len(enc.encode(prompt))
    output_tokens = len(enc.encode(text))
    
    return text, latency, input_tokens, output_tokens

# ============================================================================
# LLM-as-Judge
# ============================================================================

def judge_response_openai(client, prompt: str, expected: str, response: str) -> tuple[float, float]:
    """Use GPT-4o to judge response quality. Returns (quality_score, instruction_score)"""
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
            model="gpt-4o",
            max_tokens=256,
            messages=[{"role": "user", "content": judge_prompt}]
        )
        
        # Parse JSON from response
        text = result.choices[0].message.content
        # Handle potential markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        data = json.loads(text.strip())
        return float(data["quality"]), float(data["instruction"])
    except Exception as e:
        print(f"  ⚠️ Judge error: {e}")
        return 3.0, 3.0  # Default middle score

# ============================================================================
# Cost Calculation
# ============================================================================

def calculate_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a query."""
    config = MODELS[model_key]
    input_cost = (input_tokens / 1_000_000) * config["input_price"]
    output_cost = (output_tokens / 1_000_000) * config["output_price"]
    return input_cost + output_cost

# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_benchmark(models_to_test: list[str] = None, categories: list[str] = None):
    """Run the full benchmark suite."""
    
    if models_to_test is None:
        models_to_test = ["claude-sonnet", "gpt-4o-mini", "gemini-2.5-pro"]
    
    if categories is None:
        categories = ["clawdbot", "coding", "general"]
    
    print("=" * 60)
    print("🧪 Peter's Personal LLM Benchmark")
    print("=" * 60)
    print(f"Models: {', '.join(models_to_test)}")
    print(f"Categories: {', '.join(categories)}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Initialize clients
    clients = {}
    try:
        clients["anthropic"] = get_anthropic_client()
        print("✅ Anthropic client ready")
    except Exception as e:
        print(f"❌ Anthropic: {e}")
    
    try:
        clients["openai"] = get_openai_client()
        print("✅ OpenAI client ready")
    except Exception as e:
        print(f"❌ OpenAI: {e}")
    
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
                            clients["anthropic"], config["model"], prompt
                        )
                    elif provider == "openai":
                        response, latency, in_tok, out_tok = run_openai(
                            clients["openai"], config["model"], prompt
                        )
                    elif provider == "google":
                        response, latency, in_tok, out_tok = run_google(
                            config["model"], prompt
                        )
                    else:
                        raise ValueError(f"Unknown provider: {provider}")
                    
                    # Calculate cost
                    cost = calculate_cost(model_key, in_tok, out_tok)
                    
                    # Judge quality (using GPT-4o)
                    if "openai" in clients:
                        quality, instruction = judge_response_openai(
                            clients["openai"], prompt, expected, response
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
                        cost_usd=cost,
                        quality_score=quality,
                        instruction_score=instruction,
                    )
                    
                    print(f"✅ Q:{quality:.1f} I:{instruction:.1f} ${cost:.4f} {latency:.0f}ms")
                    
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
    """Generate markdown report from results."""
    
    report = []
    report.append("# 🧪 Peter's Personal LLM Benchmark Results")
    report.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Tests Run:** {len(results)}")
    
    # Group by model
    models = {}
    for r in results:
        if r.model not in models:
            models[r.model] = []
        models[r.model].append(r)
    
    # Summary table
    report.append("\n## 📊 Summary\n")
    report.append("| Model | Avg Quality | Avg Instruction | Avg Latency | Total Cost |")
    report.append("|-------|-------------|-----------------|-------------|------------|")
    
    model_scores = {}
    for model, model_results in models.items():
        valid = [r for r in model_results if r.error is None]
        if not valid:
            continue
        
        avg_quality = sum(r.quality_score or 0 for r in valid) / len(valid)
        avg_instruction = sum(r.instruction_score or 0 for r in valid) / len(valid)
        avg_latency = sum(r.latency_ms for r in valid) / len(valid)
        total_cost = sum(r.cost_usd for r in valid)
        
        model_scores[model] = {
            "quality": avg_quality,
            "instruction": avg_instruction,
            "latency": avg_latency,
            "cost": total_cost,
        }
        
        report.append(f"| {model} | {avg_quality:.2f}/5 | {avg_instruction:.2f}/5 | {avg_latency:.0f}ms | ${total_cost:.4f} |")
    
    # Category breakdown
    report.append("\n## 📂 By Category\n")
    
    for category in ["clawdbot", "coding", "general"]:
        report.append(f"\n### {category.title()}\n")
        report.append("| Model | Quality | Instruction | Cost |")
        report.append("|-------|---------|-------------|------|")
        
        for model in models:
            cat_results = [r for r in models[model] if r.category == category and r.error is None]
            if not cat_results:
                continue
            
            avg_q = sum(r.quality_score or 0 for r in cat_results) / len(cat_results)
            avg_i = sum(r.instruction_score or 0 for r in cat_results) / len(cat_results)
            cost = sum(r.cost_usd for r in cat_results)
            
            report.append(f"| {model} | {avg_q:.2f}/5 | {avg_i:.2f}/5 | ${cost:.4f} |")
    
    # Subscription ROI Analysis
    report.append("\n## 💰 Subscription ROI Analysis\n")
    report.append("Based on this benchmark run (extrapolated to monthly usage):\n")
    
    # Estimate monthly usage (assume 30 queries/day = 900/month)
    queries_per_month = 900
    tests_per_model = len(results) // len(models) if models else 1
    
    for sub_name, sub_info in SUBSCRIPTIONS.items():
        report.append(f"\n### {sub_name.replace('_', ' ').title()} (${sub_info['cost']}/mo)\n")
        
        for model in sub_info["models"]:
            if model in model_scores:
                scores = model_scores[model]
                cost_per_query = scores["cost"] / tests_per_model if tests_per_model > 0 else 0
                monthly_api_cost = cost_per_query * queries_per_month
                savings = monthly_api_cost - sub_info["cost"]
                
                if savings > 0:
                    verdict = f"✅ Subscription saves ${savings:.2f}/mo"
                else:
                    verdict = f"❌ API would save ${-savings:.2f}/mo"
                
                report.append(f"- **{model}:** ${cost_per_query:.4f}/query → ${monthly_api_cost:.2f}/mo API cost")
                report.append(f"  - {verdict}")
    
    # Recommendations
    report.append("\n## 🎯 Recommendations\n")
    
    if model_scores:
        best_quality = max(model_scores.items(), key=lambda x: x[1]["quality"])
        best_value = max(model_scores.items(), key=lambda x: x[1]["quality"] / (x[1]["cost"] + 0.001))
        fastest = min(model_scores.items(), key=lambda x: x[1]["latency"])
        
        report.append(f"- **Best Quality:** {best_quality[0]} ({best_quality[1]['quality']:.2f}/5)")
        report.append(f"- **Best Value:** {best_value[0]} (quality/cost ratio)")
        report.append(f"- **Fastest:** {fastest[0]} ({fastest[1]['latency']:.0f}ms avg)")
    
    # Raw results
    report.append("\n## 📋 Detailed Results\n")
    report.append("<details>")
    report.append("<summary>Click to expand full results</summary>\n")
    
    for r in results:
        report.append(f"### {r.model} / {r.category} / {r.test_name}")
        if r.error:
            report.append(f"**Error:** {r.error}")
        else:
            report.append(f"- Quality: {r.quality_score}/5 | Instruction: {r.instruction_score}/5")
            report.append(f"- Latency: {r.latency_ms:.0f}ms | Cost: ${r.cost_usd:.4f}")
            report.append(f"- Tokens: {r.input_tokens} in / {r.output_tokens} out")
            report.append(f"\n**Response:**\n```\n{r.response}\n```\n")
    
    report.append("</details>")
    
    return "\n".join(report)

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Default to testing cheaper models for initial run
    models = sys.argv[1:] if len(sys.argv) > 1 else ["claude-sonnet", "gpt-4o-mini"]
    
    print("🚀 Starting benchmark...")
    results = run_benchmark(models_to_test=models)
    
    print("\n" + "=" * 60)
    print("📊 Generating report...")
    
    report = generate_report(results)
    
    # Save report
    report_path = os.path.expanduser("~/Documents/Code/personal-llm-benchmark/results.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"✅ Report saved to: {report_path}")
    
    # Also save raw JSON
    json_path = os.path.expanduser("~/Documents/Code/personal-llm-benchmark/results.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"✅ Raw data saved to: {json_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print(report.split("## 📂")[0])  # Print just summary
