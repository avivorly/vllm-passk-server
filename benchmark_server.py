"""Benchmark using the running vLLM server (no model loading overhead)"""
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import argparse
import json
import time
import requests
from bigcode_eval.tasks import humaneval
from bigcode_eval.tasks.custom_metrics.execute import check_correctness
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_benchmark(question_idx: int, temperature: float, n: int, output_file: str, server_url: str = "http://localhost:8000"):
    """Run benchmark using the vLLM server"""

    total_start = time.time()

    # Setup task
    TaskClass = humaneval.create_task('humaneval')
    task = TaskClass()
    dataset = task.get_dataset()

    if question_idx >= len(dataset):
        print(f"Error: question_idx {question_idx} out of range (max {len(dataset)-1})")
        return

    prompt = task.get_prompt(dataset[question_idx])
    reference = task.get_reference(dataset[question_idx])
    task_id = dataset[question_idx]['task_id']
    STOP_WORDS = task.stop_words

    print(f"Benchmark: {task_id}")
    print(f"Temperature: {temperature}, n: {n}")
    print(f"Server: {server_url}")
    print("=" * 60)

    # Generation via server
    print(f"\n[GENERATION] n={n}, T={temperature}")
    gen_start = time.time()

    response = requests.post(
        f"{server_url}/generate",
        json={
            "prompt": prompt,
            "n": n,
            "temperature": temperature,
            "max_tokens": 768,
            "stop": STOP_WORDS
        },
        timeout=300
    )

    if response.status_code != 200:
        print(f"Error: {response.text}")
        return

    data = response.json()
    gen_time = time.time() - gen_start

    total_tokens = data['total_tokens']
    gen_tps = total_tokens / gen_time

    print(f"  Time: {gen_time:.2f}s")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Gen TPS: {gen_tps:,.0f}")

    # Build completions list
    completions = []
    for c in data['completions']:
        completions.append({
            'text': c['text'],
            'tokens': c['tokens'],
            'passed': None
        })

    # Testing
    print(f"\n[TESTING] 16 workers")
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

    print(f"  Time: {test_time:.2f}s")
    print(f"  Test TPS: {test_tps:,.0f}")
    print(f"  Passed: {passed_count}/{n} ({pass_rate:.1f}%)")

    # Save results
    print(f"\n[SAVING] {output_file}")
    save_start = time.time()

    total_time = time.time() - total_start
    pipeline_time = gen_time + test_time
    pipeline_tps = total_tokens / pipeline_time

    results = {
        'task_id': task_id,
        'question_idx': question_idx,
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

    save_time = time.time() - save_start
    total_time = time.time() - total_start

    results['save_time'] = round(save_time, 2)
    results['total_time'] = round(total_time, 2)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Save time: {save_time:.2f}s")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Task: {task_id}")
    print(f"  Temperature: {temperature}")
    print(f"  Completions: {n}")
    print(f"  Passed: {passed_count} ({pass_rate:.1f}%)")
    print(f"  Total tokens: {total_tokens:,}")
    print("-" * 40)
    print("TIME BREAKDOWN:")
    print(f"  Generation:    {gen_time:>8.2f}s  ({gen_time/total_time*100:>5.1f}%)")
    print(f"  Testing:       {test_time:>8.2f}s  ({test_time/total_time*100:>5.1f}%)")
    print(f"  Saving:        {save_time:>8.2f}s  ({save_time/total_time*100:>5.1f}%)")
    print(f"  TOTAL:         {total_time:>8.2f}s")
    print("-" * 40)
    print("THROUGHPUT:")
    print(f"  Gen TPS:       {gen_tps:>10,.0f}  (generation only)")
    print(f"  Test TPS:      {test_tps:>10,.0f}  (testing only)")
    print(f"  Pipeline TPS:  {pipeline_tps:>10,.0f}  (gen + test)")
    print("-" * 40)
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark using vLLM server (no model load overhead)")
    parser.add_argument('--question', '-q', type=int, default=0, help='Question index (0-163)')
    parser.add_argument('--temperature', '-t', type=float, default=0.8, help='Temperature')
    parser.add_argument('--n', type=int, default=1000, help='Number of completions')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output file')
    parser.add_argument('--server', '-s', type=str, default='http://localhost:8000', help='Server URL')
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results_q{args.question}_t{args.temperature}_n{args.n}.json"

    run_benchmark(args.question, args.temperature, args.n, args.output, args.server)
