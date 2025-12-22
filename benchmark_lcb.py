"""Benchmark LiveCodeBench with temperature sweep using vLLM"""
import os
import sys
import json
import time
import re

# Change to LiveCodeBench directory for imports (has hardcoded relative paths)
_original_cwd = os.getcwd()
LCB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LiveCodeBench')
os.chdir(LCB_PATH)
sys.path.insert(0, LCB_PATH)

from vllm import LLM, SamplingParams
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from lcb_runner.lm_styles import LMStyle
from lcb_runner.prompts.code_generation import format_prompt_generation

# Change back to original directory
os.chdir(_original_cwd)

TEMPERATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 2.0]

# Prompt mode: "few_shot" (default LCB) or "zero_shot" (no example)
PROMPT_MODE = "zero_shot"


def format_zero_shot_prompt(problem) -> str:
    """Zero-shot prompt: only the target question, no few-shot example.

    Format matches the exact specification for reproducibility.
    """
    prompt = "You are an expert Python programmer.\n"
    prompt += "You will be given a programming problem and must generate a correct\n"
    prompt += "Python solution that matches the specification and passes all\n"
    prompt += "tests.\n\n"

    prompt += problem.question_content
    prompt += "\n\n"

    prompt += "Format:\n"
    if problem.starter_code:
        prompt += "You will use the following starter code to write the solution\n"
        prompt += "and enclose your code within backticks.\n\n"
        prompt += "```python\n"
        prompt += problem.starter_code
        prompt += "\n```\n\n"
    else:
        prompt += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n\n"
        prompt += "```python\n"
        prompt += "# YOUR CODE HERE\n"
        prompt += "```\n\n"

    prompt += "Answer:\n\n"
    return prompt

# Dataset config - use "v6" for 175 new problems only (NOT "release_v6" which is cumulative 1055)
# See: https://livecodebench.github.io/ - v6 = delta between release_v5 (880) and release_v6 (1055)
DATASET_VERSION = "v6"  # Fine-grained version tag for 175 problems

# Model and generation config (saved with each result for reproducibility)
MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 2048
# Stop words to prevent repetition pattern where model loops ```python blocks
# The model generates: ```python\ncode\n```\n```python\ncode\n``` ... forever
# These stop sequences catch the end of the first complete code block
STOP_WORDS = ["```\n```", "```\n\n", "\n\n\n"]

# vLLM config - optimized for high throughput on single RTX 4090
# Note: max_num_seqs is limited by sampler warmup memory (vocab_size × max_num_seqs)
# Qwen3-0.6B has ~151K vocab, so we need smaller max_num_seqs than Qwen2.5-0.5B (~151K vs 152K)
VLLM_CONFIG = {
    "model": MODEL_NAME,
    "trust_remote_code": True,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
    "max_model_len": 3730,  # Longest prompt (1682) + max output (2048) = 3730
    "gpu_memory_utilization": 0.95,  # Max out single GPU
    "dtype": "half",
    "enforce_eager": False,
    "max_num_seqs": 512,  # Higher batch for throughput
    "max_num_batched_tokens": 32768,  # Higher batching
    "disable_log_stats": True,
}


def extract_code(text: str) -> str:
    """Extract code from markdown code blocks.

    Takes the FIRST complete code block (with stop words, we get clean output).
    """
    # Try to find ```python ... ``` blocks
    pattern = r'```python\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()  # Return FIRST code block

    # Try ``` ... ``` without language
    pattern = r'```\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # No code block found, return as-is
    return text.strip()


def test_greedy(llm, prompt, eval_sample):
    """Test with T≈0 (greedy), n=1. Returns (passed, completion)."""
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0000001,  # Near-zero for greedy
        max_tokens=MAX_TOKENS,
        stop=STOP_WORDS if STOP_WORDS else None,
    )
    outputs = llm.generate([prompt], sampling_params)
    raw_text = outputs[0].outputs[0].text
    code = extract_code(raw_text)

    # Evaluate
    metrics, results, _ = codegen_metrics(
        samples_list=[eval_sample],
        generations_list=[[code]],
        k_list=[1],
        num_process_evaluate=1,
        timeout=6,
        debug=False,
    )
    passed = all(x == True for x in results[0][0])
    return passed, code, raw_text


def get_prompt(problem) -> str:
    """Get prompt based on PROMPT_MODE setting."""
    if PROMPT_MODE == "zero_shot":
        return format_zero_shot_prompt(problem)
    else:
        return format_prompt_generation(problem, LMStyle.GenericBase)


def run_single_question(llm, problem, question_idx, n, output_dir):
    """Run temperature sweep for a single question (model already loaded)."""

    print(f"\nQuestion {question_idx}: {problem.question_title}")
    print(f"Platform: {problem.platform.value}, Difficulty: {problem.difficulty.value}")

    # Get prompt based on PROMPT_MODE
    prompt = get_prompt(problem)
    print(f"Prompt length: {len(prompt)} chars (mode: {PROMPT_MODE})")

    # Create output directory
    question_dir = f"{output_dir}/q{question_idx}"
    os.makedirs(question_dir, exist_ok=True)

    # Get evaluation sample
    eval_sample = problem.get_evaluation_sample()

    all_results = []
    total_start = time.time()

    for temp in TEMPERATURES:
        print(f"\n{'='*60}")
        print(f"Temperature: {temp}")
        print(f"{'='*60}")

        # Generation
        print(f"[GEN] n={n}, T={temp}...")
        gen_start = time.time()

        sampling_params = SamplingParams(
            n=n,
            temperature=temp,
            max_tokens=MAX_TOKENS,
            stop=STOP_WORDS if STOP_WORDS else None,
        )

        outputs = llm.generate([prompt], sampling_params)
        gen_time = time.time() - gen_start

        # Extract completions
        completions = []
        raw_outputs = []
        total_tokens = 0
        for out in outputs[0].outputs:
            raw_text = out.text
            code = extract_code(raw_text)
            completions.append(code)
            raw_outputs.append(raw_text)
            total_tokens += len(out.token_ids)

        gen_tps = total_tokens / gen_time
        print(f"      {gen_time:.2f}s | {total_tokens:,} tokens | {gen_tps:,.0f} TPS")

        # Evaluation
        print(f"[EVAL] Testing {n} completions...")
        eval_start = time.time()

        metrics, results, metadata = codegen_metrics(
            samples_list=[eval_sample],
            generations_list=[completions],
            k_list=[1, 5, 10, 50, 100, 200, 500, 1000],
            num_process_evaluate=16,
            timeout=6,
            debug=False,
        )

        eval_time = time.time() - eval_start
        eval_tps = total_tokens / eval_time

        # Count passed
        passed_count = sum(1 for r in results[0] if all(x == True for x in r))
        pass_rate = passed_count / n * 100

        print(f"      {eval_time:.2f}s | {eval_tps:,.0f} TPS | {passed_count}/{n} ({pass_rate:.1f}%)")
        print(f"      pass@1={metrics['pass@1']*100:.1f}%, pass@10={metrics.get('pass@10', 0)*100:.1f}%")

        # Save results with ALL parameters for reproducibility
        result_data = {
            'question_idx': question_idx,
            'question_title': problem.question_title,
            'platform': problem.platform.value,
            'difficulty': problem.difficulty.value,
            'dataset_version': DATASET_VERSION,
            'prompt_mode': PROMPT_MODE,
            'model_config': VLLM_CONFIG.copy(),
            'sampling_params': {
                'temperature': temp,
                'n': n,
                'max_tokens': MAX_TOKENS,
                'stop': STOP_WORDS,
            },
            'prompt': prompt,
            'total_tokens': total_tokens,
            'gen_time': round(gen_time, 2),
            'eval_time': round(eval_time, 2),
            'gen_tps': round(gen_tps),
            'eval_tps': round(eval_tps),
            'passed': passed_count,
            'pass_rate': round(pass_rate, 1),
            'metrics': metrics,
            'completions': [
                {'raw': r, 'passed': all(x == True for x in results[0][i])}
                for i, r in enumerate(raw_outputs)
            ]
        }

        temp_str = f"{temp:.1f}".replace('.', '_')
        output_file = f"{question_dir}/t{temp_str}_n{n}.json"
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"[SAVE] {output_file}")

        all_results.append({
            'temperature': temp,
            'pass_rate': pass_rate,
            'pass@1': metrics['pass@1'],
            'gen_tps': round(gen_tps),
            'eval_tps': round(eval_tps),
        })

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {problem.question_title} (Question {question_idx})")
    print(f"{'='*80}")
    print(f"{'Temp':<8} {'Pass Rate':<12} {'pass@1':<10} {'Gen TPS':<12} {'Eval TPS':<12}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['temperature']:<8.1f} {r['pass_rate']:<12.1f} {r['pass@1']*100:<10.1f} {r['gen_tps']:<12,} {r['eval_tps']:<12,}")
    print(f"{'-'*80}")
    print(f"Total time: {total_time:.1f}s")

    # Save summary
    summary_file = f"{question_dir}/summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'question_idx': question_idx,
            'question_title': problem.question_title,
            'n': n,
            'temperatures': TEMPERATURES,
            'total_time': round(total_time, 2),
            'results': all_results,
        }, f, indent=2)

    return all_results


def run_benchmark(question_idx: int = 0, n: int = 1000, output_dir: str = "results11",
                  start_idx: int = None, end_idx: int = None, skip_greedy: bool = False):
    """Run temperature sweep benchmark for single question or range."""

    # Load dataset - v6 = 175 new problems only
    print(f"Loading LiveCodeBench dataset (version={DATASET_VERSION})...")
    dataset = load_code_generation_dataset(release_version=DATASET_VERSION)
    print(f"Loaded {len(dataset)} problems (expected 175 for v6)")

    # Create output directory (use absolute path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_original_cwd, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load model once
    print(f"\nLoading {MODEL_NAME}...")
    llm = LLM(**VLLM_CONFIG)
    print("Model loaded!")

    # Determine question range
    if start_idx is not None:
        # Range mode: run questions from start_idx to end_idx
        end_idx = end_idx if end_idx is not None else len(dataset)
        questions = list(range(start_idx, min(end_idx, len(dataset))))
    else:
        # Single question mode
        questions = [question_idx]

    if not questions:
        print("No questions to process!")
        return

    skipped = []
    benchmarked = []
    global_start = time.time()

    print(f"\n{'#'*70}")
    print(f"# BENCHMARK: Questions {questions[0]} to {questions[-1]}")
    print(f"# Prompt mode: {PROMPT_MODE}")
    print(f"# Strategy: {'Skip if greedy (T≈0) passes' if skip_greedy else 'Run all'}")
    print(f"# Stop words: {STOP_WORDS if STOP_WORDS else 'None (rely on EOS)'}")
    print(f"{'#'*70}\n")

    for q_idx in questions:
        if q_idx >= len(dataset):
            print(f"[Q{q_idx}] Out of range, skipping")
            continue

        problem = dataset[q_idx]
        prompt = get_prompt(problem)
        eval_sample = problem.get_evaluation_sample()
        question_dir = f"{output_dir}/q{q_idx}"
        os.makedirs(question_dir, exist_ok=True)

        print(f"\n[Q{q_idx}] {problem.question_title}")

        # Test greedy first if enabled
        if skip_greedy:
            print(f"  Testing greedy (T≈0, n=1)...", end=" ", flush=True)
            passed, code, raw = test_greedy(llm, prompt, eval_sample)

            if passed:
                print(f"PASSED → Skipping (easy question)")
                skipped.append(q_idx)

                # Save skip record
                with open(f"{question_dir}/skipped.json", 'w') as f:
                    json.dump({
                        'question_idx': q_idx,
                        'question_title': problem.question_title,
                        'platform': problem.platform.value,
                        'difficulty': problem.difficulty.value,
                        'dataset_version': DATASET_VERSION,
                        'prompt_mode': PROMPT_MODE,
                        'model_config': VLLM_CONFIG.copy(),
                        'sampling_params': {
                            'temperature': 0.0000001,
                            'n': 1,
                            'max_tokens': MAX_TOKENS,
                            'stop': STOP_WORDS,
                        },
                        'prompt': prompt,
                        'status': 'skipped',
                        'reason': 'passed_greedy_t0',
                        'greedy_completion': {'code': code, 'raw': raw},
                    }, f, indent=2)
                continue
            else:
                print(f"FAILED → Running full temperature benchmark...")

        benchmarked.append(q_idx)
        run_single_question(llm, problem, q_idx, n, output_dir)

    global_time = time.time() - global_start

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total questions: {len(questions)}")
    print(f"Skipped (easy):  {len(skipped)}")
    print(f"Benchmarked:     {len(benchmarked)}")
    print(f"Total time:      {global_time:.1f}s")

    # Save global summary
    summary = {
        'dataset_version': DATASET_VERSION,
        'prompt_mode': PROMPT_MODE,
        'model_config': VLLM_CONFIG.copy(),
        'n': n,
        'temperatures': TEMPERATURES,
        'stop_words': STOP_WORDS,
        'questions_range': [questions[0], questions[-1]],
        'skipped': skipped,
        'benchmarked': benchmarked,
        'total_time': round(global_time, 2),
    }
    with open(f"{output_dir}/run_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {output_dir}/run_summary.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LiveCodeBench temperature sweep benchmark")
    parser.add_argument('--question', '-q', type=int, default=0, help='Single question index (used if --start not set)')
    parser.add_argument('--n', type=int, default=1000, help='Number of completions per temperature')
    parser.add_argument('--output_dir', '-o', type=str, default='results11', help='Output directory')
    parser.add_argument('--start', type=int, default=None, help='Start question index (enables range mode)')
    parser.add_argument('--end', type=int, default=None, help='End question index (exclusive)')
    parser.add_argument('--no-skip', action='store_true', help='Disable greedy skip (run all temps for every question)')
    args = parser.parse_args()

    run_benchmark(
        question_idx=args.question,
        n=args.n,
        output_dir=args.output_dir,
        start_idx=args.start,
        end_idx=args.end,
        skip_greedy=not args.no_skip
    )
