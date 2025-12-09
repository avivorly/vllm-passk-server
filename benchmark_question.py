"""Benchmark a single HumanEval question with generation + testing"""
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import argparse
import json
import time
from vllm import LLM, SamplingParams
from bigcode_eval.tasks import humaneval
from bigcode_eval.tasks.custom_metrics.execute import check_correctness
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_benchmark(question_idx: int, temperature: float, n: int, output_file: str):
    """Run benchmark for a single question"""

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
    print("=" * 60)

    # Load model
    print("Loading model...")
    load_start = time.time()
    llm = LLM(
        model='Qwen/Qwen2.5-0.5B-Instruct',
        dtype='float16',
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.95,
        max_num_seqs=2500,
        max_num_batched_tokens=150000,
        max_model_len=2048,
    )
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Generation
    print(f"\n[GENERATION] n={n}, T={temperature}")
    params = SamplingParams(n=n, temperature=temperature, max_tokens=768, top_p=0.95, stop=STOP_WORDS)

    gen_start = time.time()
    outputs = llm.generate([prompt], params)
    gen_time = time.time() - gen_start

    total_tokens = sum(len(c.token_ids) for c in outputs[0].outputs)
    gen_tps = total_tokens / gen_time

    print(f"  Time: {gen_time:.2f}s")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Gen TPS: {gen_tps:,.0f}")

    # Build completions list
    completions = []
    for c in outputs[0].outputs:
        completions.append({
            'text': c.text,
            'tokens': len(c.token_ids),
            'passed': None  # Will be filled after testing
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
    effective_tps = total_tokens / total_time

    results = {
        'task_id': task_id,
        'question_idx': question_idx,
        'temperature': temperature,
        'n': n,
        'total_tokens': total_tokens,
        'load_time': round(load_time, 2),
        'gen_time': round(gen_time, 2),
        'test_time': round(test_time, 2),
        'pipeline_time': round(pipeline_time, 2),
        'total_time': round(total_time, 2),
        'gen_tps': round(gen_tps),
        'test_tps': round(test_tps),
        'pipeline_tps': round(pipeline_tps),
        'effective_tps': round(effective_tps),
        'passed': passed_count,
        'pass_rate': round(pass_rate, 1),
        'completions': completions
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    save_time = time.time() - save_start
    total_time = time.time() - total_start
    effective_tps = total_tokens / total_time

    # Update results with final timing
    results['save_time'] = round(save_time, 2)
    results['total_time'] = round(total_time, 2)
    results['effective_tps'] = round(effective_tps)

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
    print(f"  Model load:    {load_time:>8.2f}s  ({load_time/total_time*100:>5.1f}%)")
    print(f"  Generation:    {gen_time:>8.2f}s  ({gen_time/total_time*100:>5.1f}%)")
    print(f"  Testing:       {test_time:>8.2f}s  ({test_time/total_time*100:>5.1f}%)")
    print(f"  Saving:        {save_time:>8.2f}s  ({save_time/total_time*100:>5.1f}%)")
    print(f"  TOTAL:         {total_time:>8.2f}s")
    print("-" * 40)
    print("THROUGHPUT:")
    print(f"  Gen TPS:       {gen_tps:>10,.0f}  (generation only)")
    print(f"  Test TPS:      {test_tps:>10,.0f}  (testing only)")
    print(f"  Pipeline TPS:  {pipeline_tps:>10,.0f}  (gen + test)")
    print(f"  Effective TPS: {effective_tps:>10,.0f}  (including model load)")
    print("-" * 40)
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a single HumanEval question")
    parser.add_argument('--question', '-q', type=int, default=0, help='Question index (0-163)')
    parser.add_argument('--temperature', '-t', type=float, default=0.8, help='Temperature')
    parser.add_argument('--n', type=int, default=1000, help='Number of completions')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output file (default: results_q{q}_t{t}_n{n}.json)')
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results_q{args.question}_t{args.temperature}_n{args.n}.json"

    run_benchmark(args.question, args.temperature, args.n, args.output)
