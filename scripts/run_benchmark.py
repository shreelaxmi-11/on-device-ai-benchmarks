"""
MLX LLM Inference Benchmark
MacBook Pro M3 Pro — Apple Silicon

Usage:
    python scripts/run_benchmark.py --model mlx-community/Llama-3.2-3B-Instruct-4bit
    python scripts/run_benchmark.py --all
    python scripts/run_benchmark.py --model <hf-id> --prompt-tokens 512 --gen-tokens 256 --runs 5
"""

import argparse
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load, generate, stream_generate
except ImportError:
    print("ERROR: mlx and mlx-lm required. Install with: pip install mlx mlx-lm")
    sys.exit(1)


# ── Default model suite ──────────────────────────────────────────────────────

DEFAULT_MODELS = [
    {
        "name": "Llama 3.2 3B (INT4)",
        "hf_id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "quant": "Q4",
        "expected_size_gb": 2.1,
    },
    {
        "name": "Phi-3.5 Mini (INT4)",
        "hf_id": "mlx-community/Phi-3.5-mini-instruct-4bit",
        "quant": "Q4",
        "expected_size_gb": 2.3,
    },
    {
        "name": "Mistral 7B v0.3 (INT4)",
        "hf_id": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "quant": "Q4",
        "expected_size_gb": 4.1,
    },
    {
        "name": "Llama 3 8B (INT4)",
        "hf_id": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        "quant": "Q4",
        "expected_size_gb": 4.5,
    },
]

BENCHMARK_PROMPT = (
    "Explain the key tradeoffs between on-device AI inference and cloud-based AI inference, "
    "focusing on latency, privacy, memory constraints, and battery consumption. "
    "Provide specific examples from real mobile AI deployments like Apple Intelligence and Google Gemini Nano."
)


# ── Core benchmark function ───────────────────────────────────────────────────

def benchmark_model(hf_id: str, prompt: str, gen_tokens: int = 256, runs: int = 3) -> dict:
    """
    Load model, run N inference passes, return timing + memory stats.
    """
    print(f"\n{'─' * 60}")
    print(f"Model: {hf_id}")
    print(f"{'─' * 60}")

    # Load
    print("Loading model...")
    load_start = time.perf_counter()
    model, tokenizer = load(hf_id)
    load_time = time.perf_counter() - load_start
    print(f"Load time: {load_time:.1f}s")

    # Tokenize prompt to count tokens
    if hasattr(tokenizer, 'encode'):
        prompt_tokens = len(tokenizer.encode(prompt))
    else:
        prompt_tokens = len(prompt.split()) * 1.3  # rough estimate
    print(f"Prompt tokens: {int(prompt_tokens)}")

    # Memory after load
    mem_gb = _get_memory_gb()
    print(f"Model memory: ~{mem_gb:.1f} GB")

    results_per_run = []

    for i in range(runs):
        print(f"\nRun {i+1}/{runs}...")

        # Time to first token
        ttft_start = time.perf_counter()
        first_token_received = False
        tokens_generated = 0
        gen_start = None

        # Use generate() for clean timing
        gen_start = time.perf_counter()

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=gen_tokens,
            verbose=False,
        )

        gen_end = time.perf_counter()
        total_time = gen_end - gen_start

        # Count output tokens
        if hasattr(tokenizer, 'encode'):
            out_tokens = len(tokenizer.encode(response))
        else:
            out_tokens = len(response.split()) * 1.3

        gen_tok_per_sec = out_tokens / total_time if total_time > 0 else 0

        run_result = {
            "run": i + 1,
            "total_time_s": round(total_time, 3),
            "output_tokens": int(out_tokens),
            "gen_tok_per_sec": round(gen_tok_per_sec, 1),
        }
        results_per_run.append(run_result)
        print(f"  Gen speed: {gen_tok_per_sec:.1f} tok/s | {out_tokens} tokens in {total_time:.2f}s")

    # Aggregate
    avg_gen_tps = sum(r["gen_tok_per_sec"] for r in results_per_run) / runs
    avg_time = sum(r["total_time_s"] for r in results_per_run) / runs

    result = {
        "model": hf_id,
        "prompt_tokens": int(prompt_tokens),
        "gen_tokens_requested": gen_tokens,
        "runs": runs,
        "memory_gb": round(mem_gb, 1),
        "load_time_s": round(load_time, 1),
        "avg_gen_tok_per_sec": round(avg_gen_tps, 1),
        "avg_time_s": round(avg_time, 2),
        "per_run": results_per_run,
    }

    print(f"\n✓ Average: {avg_gen_tps:.1f} tok/s | Memory: {mem_gb:.1f} GB")

    # Free model from memory
    del model
    mx.clear_cache() if hasattr(mx, 'metal') else None

    return result


def _get_memory_gb() -> float:
    """Approximate GPU/unified memory used via mx."""
    try:
        mem = mx.get_peak_memory() if hasattr(mx, 'metal') else 0
        return mem / (1024 ** 3)
    except Exception:
        return 0.0


# ── Output formatting ─────────────────────────────────────────────────────────

def results_to_markdown(results: list, hardware_info: dict) -> str:
    header = f"""# MLX Benchmark Results

**Hardware:** {hardware_info.get('chip', 'Apple Silicon')} · {hardware_info.get('memory', '?')} Unified Memory
**Date:** {datetime.now().strftime('%Y-%m-%d')}
**MLX version:** {_get_mlx_version()}

## Results

| Model | Memory (GB) | Gen Speed (tok/s) | Avg Time (s) |
|-------|------------|-------------------|-------------|
"""
    rows = ""
    for r in results:
        model_short = r["model"].split("/")[-1]
        rows += f"| {model_short} | {r['memory_gb']} | **{r['avg_gen_tok_per_sec']}** | {r['avg_time_s']} |\n"

    footer = f"""
**Prompt:** {results[0]['prompt_tokens'] if results else '?'} tokens
**Generation:** {results[0]['gen_tokens_requested'] if results else '?'} tokens
**Runs per model:** {results[0]['runs'] if results else '?'}
"""
    return header + rows + footer


def _get_mlx_version() -> str:
    try:
        import mlx
        return getattr(mlx, '__version__', 'unknown')
    except Exception:
        return 'unknown'


def get_hardware_info() -> dict:
    info = {}
    try:
        import subprocess
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout
        for line in lines.split('\n'):
            if 'Chip' in line:
                info['chip'] = line.split(':')[-1].strip()
            if 'Memory' in line and 'GB' in line:
                info['memory'] = line.split(':')[-1].strip()
    except Exception:
        info['chip'] = 'Apple Silicon'
        info['memory'] = '18 GB'
    return info


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLX LLM Inference Benchmark")
    parser.add_argument("--model", type=str, help="HuggingFace model ID (e.g., mlx-community/Llama-3.2-3B-Instruct-4bit)")
    parser.add_argument("--all", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--prompt-tokens", type=int, default=None, help="Use a prompt of approximately N tokens")
    parser.add_argument("--gen-tokens", type=int, default=256, help="Tokens to generate per run (default: 256)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per model (default: 3)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results (default: results/)")
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.print_help()
        print("\nExample: python scripts/run_benchmark.py --model mlx-community/Llama-3.2-3B-Instruct-4bit")
        sys.exit(0)

    hw = get_hardware_info()
    print(f"\n{'═' * 60}")
    print(f"  MLX Benchmark — {hw.get('chip', 'Apple Silicon')}")
    print(f"  Memory: {hw.get('memory', '?')} | MLX {_get_mlx_version()}")
    print(f"  Device: {mx.default_device()}")
    print(f"{'═' * 60}")

    prompt = BENCHMARK_PROMPT

    models_to_run = []
    if args.all:
        models_to_run = [m["hf_id"] for m in DEFAULT_MODELS]
    elif args.model:
        models_to_run = [args.model]

    all_results = []
    for model_id in models_to_run:
        try:
            result = benchmark_model(
                hf_id=model_id,
                prompt=prompt,
                gen_tokens=args.gen_tokens,
                runs=args.runs,
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR running {model_id}: {e}")

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({"hardware": hw, "results": all_results}, f, indent=2)
    print(f"\n✓ JSON results saved: {json_path}")

    md_path = out_dir / f"benchmark_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(results_to_markdown(all_results, hw))
    print(f"✓ Markdown results saved: {md_path}")

    # Print summary
    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    print(f"{'═' * 60}")
    print(f"{'Model':<40} {'Gen (tok/s)':>12} {'Memory (GB)':>12}")
    print(f"{'─' * 64}")
    for r in all_results:
        name = r["model"].split("/")[-1][:38]
        print(f"{name:<40} {r['avg_gen_tok_per_sec']:>12.1f} {r['memory_gb']:>12.1f}")


if __name__ == "__main__":
    main()
