"""Benchmark a question across multiple temperatures"""
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import argparse
import json
import time
import requests
from bigcode_eval.tasks import humaneval
from bigcode_eval.tasks.custom_metrics.execute import check_correctness
from concurrent.futures import ProcessPoolExecutor, as_completed


TEMPERATURES = [0.0000001, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0]


def run_single_benchmark(question_idx: int, temperature: float, n: int, output_file: str,
                         prompt: str, reference: str, task_id: str, stop_words: list,
                         server_url: str = "http://localhost:8000"):
    """Run benchmark for a single temperature"""

    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"T={temperature} | {task_id}")
    print(f"{'='*60}")

    # Generation via server
    print(f"[GEN] n={n}, T={temperature}...")
    gen_start = time.time()

    response = requests.post(
        f"{server_url}/generate",
        json={
            "prompt": prompt,
            "n": n,
            "temperature": temperature,
            "max_tokens": 768,
            "stop": stop_words
        },
        timeout=300
    )

    if response.status_code != 200:
        print(f"Error: {response.text}")
        return None

    data = response.json()
    gen_time = time.time() - gen_start

    total_tokens = data['total_tokens']
    gen_tps = total_tokens / gen_time

    print(f"      {gen_time:.2f}s | {total_tokens:,} tokens | {gen_tps:,.0f} TPS")

    # Build completions list
    completions = []
    for c in data['completions']:
        completions.append({
            'text': c['text'],
            'tokens': c['tokens'],
            'passed': None
        })

    # Testing
    print(f"[TEST] 16 workers...")
    test_start = time.time()

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {}
        for i, comp in enumerate(completions):
            test_program = prompt + comp['text'] + '\n' + reference
            future = executor.submit(check_correctness, test_program, 3.0, 0, i)
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            completions[idx]['passed'] = result['passed']

    test_time = time.time() - test_start
    test_tps = total_tokens / test_time

    passed_count = sum(1 for c in completions if c['passed'])
    pass_rate = passed_count / n * 100

    print(f"      {test_time:.2f}s | {test_tps:,.0f} TPS | {passed_count}/{n} ({pass_rate:.1f}%)")

    # Save results
    total_time = time.time() - total_start
    pipeline_time = gen_time + test_time
    pipeline_tps = total_tokens / pipeline_time

    results = {
        'task_id': task_id,
        'question_idx': question_idx,
        # Reproducibility: save exact prompt and all params
        'prompt': prompt,
        'reference': reference,
        'params': {
            'temperature': temperature,
            'n': n,
            'max_tokens': 768,
            'stop': stop_words
        },
        # Results
        'temperature': temperature,
        'n': n,
        'total_tokens': total_tokens,
        'gen_time': round(gen_time, 2),
        'test_time': round(test_time, 2),
        'pipeline_time': round(pipeline_time, 2),
        'total_time': round(total_time, 2),
        'gen_tps': round(gen_tps),
        'test_tps': round(test_tps),
        'pipeline_tps': round(pipeline_tps),
        'passed': passed_count,
        'pass_rate': round(pass_rate, 1),
        'completions': completions
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[SAVE] {output_file}")

    return results


def benchmark_question_all_temperatures(question_idx: int, n: int = 1000,
                                        output_dir: str = "results",
                                        server_url: str = "http://localhost:8000",
                                        use_stop_words: bool = True):
    """Run benchmark for a single question across all temperatures"""

    # Setup task
    TaskClass = humaneval.create_task('humaneval')
    task = TaskClass()
    dataset = task.get_dataset()

    if question_idx >= len(dataset):
        print(f"Error: question_idx {question_idx} out of range (max {len(dataset)-1})")
        return

    prompt = task.get_prompt(dataset[question_idx]) + "\n"
    reference = task.get_reference(dataset[question_idx])
    task_id = dataset[question_idx]['task_id']
    stop_words = task.stop_words if use_stop_words else []

    # Create question-specific directory
    question_dir = f"{output_dir}/q{question_idx}"
    os.makedirs(question_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# Benchmark: {task_id} (Question {question_idx})")
    print(f"# n={n}, Temperatures: {len(TEMPERATURES)}")
    print(f"# Output: {question_dir}/")
    print(f"{'#'*60}")

    all_results = []
    total_start = time.time()

    for temp in TEMPERATURES:
        # Format temperature for filename
        temp_str = f"{temp:.7f}".rstrip('0').rstrip('.')
        output_file = f"{question_dir}/t{temp_str}_n{n}.json"

        result = run_single_benchmark(
            question_idx=question_idx,
            temperature=temp,
            n=n,
            output_file=output_file,
            prompt=prompt,
            reference=reference,
            task_id=task_id,
            stop_words=stop_words,
            server_url=server_url
        )

        if result:
            all_results.append({
                'temperature': temp,
                'gen_tps': result['gen_tps'],
                'test_tps': result['test_tps'],
                'pipeline_tps': result['pipeline_tps'],
                'pass_rate': result['pass_rate'],
                'passed': result['passed']
            })

    total_time = time.time() - total_start

    # Print summary table
    print(f"\n{'='*90}")
    print(f"SUMMARY: {task_id} (Question {question_idx})")
    print(f"{'='*90}")
    print(f"{'Temp':<12} {'Gen TPS':<12} {'Test TPS':<12} {'Pipeline TPS':<14} {'Pass Rate':<12} {'Passed':<8}")
    print(f"{'-'*90}")

    for r in all_results:
        print(f"{r['temperature']:<12.7g} {r['gen_tps']:<12,} {r['test_tps']:<12,} {r['pipeline_tps']:<14,} {r['pass_rate']:<12.1f} {r['passed']:<8}")

    print(f"{'-'*70}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results saved to: {question_dir}/")

    # Save summary inside question directory
    summary_file = f"{question_dir}/summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'task_id': task_id,
            'question_idx': question_idx,
            'n': n,
            'total_time': round(total_time, 2),
            'results': all_results
        }, f, indent=2)
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a question across all temperatures")
    parser.add_argument('--question', '-q', type=int, default=0, help='Question index (0-163)')
    parser.add_argument('--n', type=int, default=1000, help='Number of completions per temperature')
    parser.add_argument('--output_dir', '-o', type=str, default='results', help='Output directory')
    parser.add_argument('--server', '-s', type=str, default='http://localhost:8000', help='Server URL')
    parser.add_argument('--no-stop', action='store_true', help='Disable stop words (let model generate freely)')
    args = parser.parse_args()

    benchmark_question_all_temperatures(args.question, args.n, args.output_dir, args.server,
                                        use_stop_words=not args.no_stop)
