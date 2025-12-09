"""Benchmark all questions - skip easy ones that pass with greedy decoding"""
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import argparse
import json
import time
import requests
from bigcode_eval.tasks import humaneval
from bigcode_eval.tasks.custom_metrics.execute import check_correctness
from benchmark_temperatures import benchmark_question_all_temperatures

# Setup task globally
TaskClass = humaneval.create_task('humaneval')
task = TaskClass()
dataset = task.get_dataset()
STOP_WORDS = task.stop_words


def test_greedy(question_idx: int, server_url: str = "http://localhost:8000", use_stop_words: bool = True):
    """Test a question with T=0 (greedy) and n=1. Returns True if passed."""

    prompt = task.get_prompt(dataset[question_idx])
    reference = task.get_reference(dataset[question_idx])
    task_id = dataset[question_idx]['task_id']
    stop_words = STOP_WORDS if use_stop_words else []

    # Generate with T≈0 (greedy)
    response = requests.post(
        f"{server_url}/generate",
        json={
            "prompt": prompt,
            "n": 1,
            "temperature": 0.0000001,  # Near-zero for greedy
            "max_tokens": 768,
            "stop": stop_words
        },
        timeout=60
    )

    if response.status_code != 200:
        print(f"  Error: {response.text}")
        return False

    data = response.json()
    completion_text = data['completions'][0]['text']

    # Test the completion
    test_program = prompt + completion_text + '\n' + reference
    result = check_correctness(test_program, 3.0, 0, 0)

    return result['passed']


def benchmark_all_questions(start_idx: int = 0, end_idx: int = 164, n: int = 1000,
                            output_dir: str = "results", server_url: str = "http://localhost:8000",
                            use_stop_words: bool = True):
    """
    Loop over all questions:
    - First test with T=0, n=1 (greedy)
    - If passed: skip (easy question)
    - If failed: run full temperature benchmark
    """

    os.makedirs(output_dir, exist_ok=True)

    skipped = []
    benchmarked = []
    total_start = time.time()

    stop_status = "ON" if use_stop_words else "OFF"
    print(f"\n{'#'*70}")
    print(f"# BENCHMARK ALL QUESTIONS: {start_idx} to {end_idx-1}")
    print(f"# Strategy: Skip if greedy (T=0) passes, else run all temperatures")
    print(f"# Stop words: {stop_status}")
    print(f"{'#'*70}\n")

    for q_idx in range(start_idx, end_idx):
        task_id = dataset[q_idx]['task_id']

        print(f"\n[Q{q_idx}] {task_id}")
        print(f"  Testing greedy (T=0, n=1)...", end=" ", flush=True)

        passed_greedy = test_greedy(q_idx, server_url, use_stop_words)

        if passed_greedy:
            print(f"PASSED → Skipping (easy question)")
            skipped.append(q_idx)

            # Save skip record with full reproducibility info in question directory
            prompt = task.get_prompt(dataset[q_idx])
            reference = task.get_reference(dataset[q_idx])

            question_dir = f"{output_dir}/q{q_idx}"
            os.makedirs(question_dir, exist_ok=True)

            skip_file = f"{question_dir}/skipped.json"
            with open(skip_file, 'w') as f:
                json.dump({
                    'task_id': task_id,
                    'question_idx': q_idx,
                    'prompt': prompt,
                    'reference': reference,
                    'params': {
                        'temperature': 0.0000001,
                        'n': 1,
                        'max_tokens': 768,
                        'stop': STOP_WORDS
                    },
                    'status': 'skipped',
                    'reason': 'passed_greedy_t0'
                }, f, indent=2)
        else:
            print(f"FAILED → Running full temperature benchmark...")
            benchmarked.append(q_idx)

            # Run full temperature benchmark
            benchmark_question_all_temperatures(
                question_idx=q_idx,
                n=n,
                output_dir=output_dir,
                server_url=server_url,
                use_stop_words=use_stop_words
            )

    total_time = time.time() - total_start

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total questions: {end_idx - start_idx}")
    print(f"Skipped (easy):  {len(skipped)} - {skipped}")
    print(f"Benchmarked:     {len(benchmarked)} - {benchmarked}")
    print(f"Total time:      {total_time:.1f}s")

    # Save global summary file
    global_summary = {
        'run_info': {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'n': n,
            'total_questions': end_idx - start_idx,
            'total_time': round(total_time, 2)
        },
        'skipped': skipped,
        'benchmarked': benchmarked,
        'questions': {}
    }

    # Load each question's summary
    for q_idx in range(start_idx, end_idx):
        question_dir = f"{output_dir}/q{q_idx}"
        summary_file = f"{question_dir}/summary.json"
        skipped_file = f"{question_dir}/skipped.json"

        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                global_summary['questions'][q_idx] = json.load(f)
        elif os.path.exists(skipped_file):
            with open(skipped_file, 'r') as f:
                data = json.load(f)
                global_summary['questions'][q_idx] = {
                    'task_id': data['task_id'],
                    'status': 'skipped',
                    'reason': data['reason']
                }

    # Save global summary
    global_summary_file = f"{output_dir}/run_summary_q{start_idx}-q{end_idx-1}.json"
    with open(global_summary_file, 'w') as f:
        json.dump(global_summary, f, indent=2)
    print(f"Global summary:  {global_summary_file}")

    return skipped, benchmarked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark all questions with smart skipping")
    parser.add_argument('--start', type=int, default=0, help='Start question index')
    parser.add_argument('--end', type=int, default=164, help='End question index (exclusive)')
    parser.add_argument('--n', type=int, default=1000, help='Number of completions per temperature')
    parser.add_argument('--output_dir', '-o', type=str, default='results', help='Output directory')
    parser.add_argument('--server', '-s', type=str, default='http://localhost:8000', help='Server URL')
    parser.add_argument('--no-stop', action='store_true', help='Disable stop words (let model generate freely)')
    args = parser.parse_args()

    benchmark_all_questions(args.start, args.end, args.n, args.output_dir, args.server,
                           use_stop_words=not args.no_stop)
