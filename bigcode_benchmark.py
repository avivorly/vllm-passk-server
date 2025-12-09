"""BigCode Evaluation Harness benchmark: Generation + Testing TPS"""
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import time
from vllm import LLM, SamplingParams
from bigcode_eval.tasks import humaneval
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval
from concurrent.futures import ProcessPoolExecutor, as_completed
from bigcode_eval.tasks.custom_metrics.execute import check_correctness
from collections import Counter, defaultdict
import numpy as np

# Create humaneval task
TaskClass = humaneval.create_task('humaneval')
task = TaskClass()
dataset = task.get_dataset()
STOP_WORDS = task.stop_words

def run_benchmark(n_values=[100, 1000], test_workers=16, max_tokens=768, temperature=0.8):
    """Run full generation + testing benchmark"""

    # Initialize vLLM with optimized config
    print("Loading vLLM model...")
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
    print("Model loaded!\n")

    # Get all prompts and references (test code + entry point call)
    prompts = [task.get_prompt(dataset[i]) for i in range(len(dataset))]
    references = [task.get_reference(dataset[i]) for i in range(len(dataset))]

    print(f"BigCode HumanEval benchmark")
    print(f"Problems: {len(prompts)}")
    print(f"Stop words: {STOP_WORDS}")
    print(f"Max tokens: {max_tokens}, Temperature: {temperature}")
    print(f"Test workers: {test_workers}")
    print("=" * 70)

    for n in n_values:
        print(f"\n{'='*70}")
        print(f"n = {n} completions per problem")
        print(f"{'='*70}")

        # ===== GENERATION PHASE =====
        print("\n[GENERATION PHASE]")
        params = SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            stop=STOP_WORDS,
        )

        gen_start = time.time()
        outputs = llm.generate(prompts, params)
        gen_time = time.time() - gen_start

        # Count tokens and prepare predictions
        total_tokens = 0
        predictions = []

        for i, output in enumerate(outputs):
            prompt = prompts[i]
            completions = []
            for c in output.outputs:
                total_tokens += len(c.token_ids)
                # Postprocess: prompt + completion text
                full_code = prompt + c.text
                completions.append(full_code)
            predictions.append(completions)

        gen_tps = total_tokens / gen_time
        avg_tokens = total_tokens / (len(prompts) * n)

        print(f"  Time: {gen_time:.2f}s")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Avg tokens/completion: {avg_tokens:.1f}")
        print(f"  Generation TPS: {gen_tps:,.0f}")

        # ===== TESTING PHASE =====
        print("\n[TESTING PHASE]")

        # Use custom parallel testing with ProcessPoolExecutor for better performance
        test_start = time.time()

        # Build test programs
        test_args = []
        completion_id = Counter()
        for task_id, (candidates, test_case) in enumerate(zip(predictions, references)):
            for candidate in candidates:
                test_program = candidate + "\n" + test_case
                args = (test_program, 3.0, task_id, completion_id[task_id])
                test_args.append(args)
                completion_id[task_id] += 1

        # Run tests in parallel with ProcessPoolExecutor
        results = defaultdict(list)
        with ProcessPoolExecutor(max_workers=test_workers) as executor:
            futures = []
            for args in test_args:
                future = executor.submit(check_correctness, *args)
                futures.append(future)

            for future in as_completed(futures):
                result = future.result()
                results[result["task_id"]].append((result["completion_id"], result))

        test_time = time.time() - test_start
        test_tps = total_tokens / test_time

        # Calculate pass@k
        total, correct = [], []
        for result in results.values():
            result.sort()
            passed = [r[1]["passed"] for r in result]
            total.append(len(passed))
            correct.append(sum(passed))
        total_arr = np.array(total)
        correct_arr = np.array(correct)

        def estimate_pass_at_k(num_samples, num_correct, k):
            def estimator(n, c, k):
                if n - c < k:
                    return 1.0
                return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
            return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct)])

        pass_at_1 = estimate_pass_at_k(total_arr, correct_arr, 1).mean() * 100
        pass_at_10 = estimate_pass_at_k(total_arr, correct_arr, min(10, n)).mean() * 100 if n >= 10 else 0
        pass_at_100 = estimate_pass_at_k(total_arr, correct_arr, min(100, n)).mean() * 100 if n >= 100 else 0

        print(f"  Time: {test_time:.2f}s ({test_workers} workers)")
        print(f"  Tests executed: {len(test_args):,}")
        print(f"  Test TPS: {test_tps:,.0f}")
        print(f"  pass@1: {pass_at_1:.1f}%")
        if n >= 10:
            print(f"  pass@10: {pass_at_10:.1f}%")
        if n >= 100:
            print(f"  pass@100: {pass_at_100:.1f}%")

        # ===== SUMMARY =====
        total_time = gen_time + test_time
        print(f"\n[SUMMARY for n={n}]")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Generation: {gen_time:.2f}s ({gen_time/total_time*100:.0f}%)")
        print(f"  Testing: {test_time:.2f}s ({test_time/total_time*100:.0f}%)")
        print(f"  End-to-end TPS: {total_tokens/total_time:,.0f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='+', default=[100, 1000])
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--max_tokens', type=int, default=768)
    parser.add_argument('--temperature', type=float, default=0.8)
    args = parser.parse_args()

    run_benchmark(
        n_values=args.n,
        test_workers=args.workers,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
