#!/usr/bin/env python3
"""
Async benchmark: GPUs generate continuously, CPU evaluates in parallel.
GPU never waits for eval - immediately starts next generation.
"""
import os
import sys
import json
import time
import subprocess
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Lock
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

TEMPERATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 2.0]
NUM_GPUS = 8
NUM_EVAL_WORKERS = 24  # CPU workers for evaluation


def evaluate_file(gen_file):
    """Evaluate a single generation file - runs in separate process"""
    import os
    import sys
    import json

    os.chdir('/workspace/orkspace/LiveCodeBench')
    sys.path.insert(0, '/workspace/orkspace/LiveCodeBench')

    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    try:
        with open(gen_file) as f:
            data = json.load(f)

        if data.get('evaluated', False):
            return gen_file, data  # Already evaluated

        # Load dataset for eval_sample
        dataset = load_code_generation_dataset(release_version="v6")
        problem = dataset[data['question_idx']]
        eval_sample = problem.get_evaluation_sample()

        completions = data['completions']

        eval_start = time.time()
        metrics, results, _ = codegen_metrics(
            samples_list=[eval_sample],
            generations_list=[completions],
            k_list=[1],
            num_process_evaluate=1,  # Single process since we're already parallel
            timeout=6,
            debug=False,
        )
        eval_time = time.time() - eval_start

        passed_count = sum(1 for r in results[0] if all(x == True for x in r))
        pass_list = [all(x == True for x in r) for r in results[0]]

        # Update data with eval results
        data['eval_time'] = eval_time
        data['eval_tps'] = data['n'] / eval_time if eval_time > 0 else 0
        data['passed'] = passed_count
        data['pass_rate'] = passed_count / data['n'] * 100
        data['evaluated'] = True
        data['completions'] = [
            {"raw": raw, "passed": passed}
            for raw, passed in zip(data['raw_outputs'], pass_list)
        ]
        del data['raw_outputs']  # Remove raw to save space

        # Save evaluated results
        with open(gen_file, 'w') as f:
            json.dump(data, f)

        return gen_file, data

    except Exception as e:
        return gen_file, {"error": str(e)}


def gpu_generator_thread(gpu_id, gen_queue, eval_queue, output_dir):
    """GPU thread: generates and immediately queues for eval, grabs next job"""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['HF_HOME'] = '/workspace/.cache/huggingface'

    while True:
        try:
            job = gen_queue.get(timeout=1)
        except Empty:
            continue

        if job is None:  # Poison pill
            break

        question_idx, temp, n = job
        temp_str = f"{temp:.1f}".replace('.', '_')
        output_file = os.path.join(output_dir, f"gpu{gpu_id}_t{temp_str}_gen.json")

        cmd = [
            'python3', '/workspace/orkspace/gpu_gen_only.py',
            str(gpu_id), str(question_idx), str(n), str(temp), output_file
        ]

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(output_file):
            # Queue for evaluation immediately
            eval_queue.put((output_file, temp))
        else:
            print(f"  [GPU{gpu_id}] Gen T={temp} failed", file=sys.stderr)

        gen_queue.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=175)
    parser.add_argument('--n', type=int, default=3000)
    parser.add_argument('--output_dir', '-o', type=str, default='results_async')
    parser.add_argument('--temps', type=str, default=None)
    args = parser.parse_args()

    temps = [float(t) for t in args.temps.split(',')] if args.temps else TEMPERATURES

    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    sys.path.insert(0, '/workspace/orkspace/LiveCodeBench')
    os.chdir('/workspace/orkspace/LiveCodeBench')
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    dataset = load_code_generation_dataset(release_version="v6")
    os.chdir('/workspace/orkspace')

    print(f"{'='*70}")
    print(f"ASYNC BENCHMARK: GPU gen + CPU eval in parallel")
    print(f"{args.end - args.start} questions x {len(temps)} temps x {args.n} completions")
    print(f"{NUM_GPUS} GPUs generating, {NUM_EVAL_WORKERS} CPU workers evaluating")
    print(f"{'='*70}")

    global_start = time.time()

    for q_idx in range(args.start, min(args.end, len(dataset))):
        problem = dataset[q_idx]
        q_dir = output_dir / f"q{q_idx}"
        q_dir.mkdir(exist_ok=True)

        print(f"\n[Q{q_idx}] {problem.question_title}")
        q_start = time.time()

        # Queues
        gen_queue = Queue()
        eval_queue = Queue()

        # Add generation jobs - split each temp across GPUs
        n_per_gpu = args.n // NUM_GPUS
        for temp in temps:
            for _ in range(NUM_GPUS):
                gen_queue.put((q_idx, temp, n_per_gpu))

        # Start GPU generator threads
        gpu_threads = []
        for gpu_id in range(NUM_GPUS):
            t = Thread(target=gpu_generator_thread,
                      args=(gpu_id, gen_queue, eval_queue, str(q_dir)))
            t.start()
            gpu_threads.append(t)
            time.sleep(0.5)  # Slight stagger

        # Collect generated files and evaluate with process pool
        results_by_temp = {temp: [] for temp in temps}
        total_gen_jobs = len(temps) * NUM_GPUS
        completed_gen = 0
        completed_eval = 0

        with ProcessPoolExecutor(max_workers=NUM_EVAL_WORKERS) as executor:
            eval_futures = {}

            while completed_eval < total_gen_jobs:
                # Submit new eval jobs
                try:
                    while True:
                        gen_file, temp = eval_queue.get_nowait()
                        future = executor.submit(evaluate_file, gen_file)
                        eval_futures[future] = temp
                        completed_gen += 1
                except Empty:
                    pass

                # Check completed evals
                done_futures = [f for f in eval_futures if f.done()]
                for future in done_futures:
                    temp = eval_futures.pop(future)
                    try:
                        gen_file, result = future.result()
                        if 'error' not in result:
                            results_by_temp[temp].append(result)
                        completed_eval += 1
                    except Exception as e:
                        print(f"  Eval error: {e}", file=sys.stderr)
                        completed_eval += 1

                time.sleep(0.1)

        # Send poison pills to stop GPU threads
        for _ in range(NUM_GPUS):
            gen_queue.put(None)
        for t in gpu_threads:
            t.join()

        # Combine results per temperature and save
        q_results = []
        for temp in temps:
            gpu_results = results_by_temp[temp]
            if not gpu_results:
                continue

            # Combine
            total_tokens = sum(r['total_tokens'] for r in gpu_results)
            max_gen_time = max(r['gen_time'] for r in gpu_results)
            max_eval_time = max(r.get('eval_time', 0) for r in gpu_results)
            total_passed = sum(r.get('passed', 0) for r in gpu_results)
            total_n = sum(r['n'] for r in gpu_results)

            combined = {
                'question_idx': q_idx,
                'question_title': problem.question_title,
                'temperature': temp,
                'n': total_n,
                'total_tokens': total_tokens,
                'gen_time': max_gen_time,
                'gen_tps': total_tokens / max_gen_time if max_gen_time > 0 else 0,
                'eval_time': max_eval_time,
                'eval_tps': total_n / max_eval_time if max_eval_time > 0 else 0,
                'passed': total_passed,
                'pass_rate': total_passed / total_n * 100 if total_n > 0 else 0,
                'model': gpu_results[0].get('model'),
                'sampling_params': gpu_results[0].get('sampling_params'),
                'model_config': gpu_results[0].get('model_config'),
                'prompt': gpu_results[0].get('prompt'),
                'completions': [c for r in gpu_results for c in r.get('completions', [])],
            }

            temp_str = f"{temp:.1f}".replace('.', '_')
            with open(q_dir / f"t{temp_str}_n{args.n}.json", 'w') as f:
                json.dump(combined, f, indent=2)

            q_results.append(combined)
            print(f"  T={temp:.1f}: gen:{combined['gen_tps']:,.0f} eval:{combined['eval_tps']:,.0f} | {combined['passed']}/{combined['n']} ({combined['pass_rate']:.1f}%)")

            # Cleanup temp files
            for gpu_id in range(NUM_GPUS):
                gf = q_dir / f"gpu{gpu_id}_t{temp_str}_gen.json"
                if gf.exists():
                    gf.unlink()

        q_time = time.time() - q_start

        # Save summary
        summary = {
            'question_idx': q_idx,
            'question_title': problem.question_title,
            'n': args.n,
            'temperatures': temps,
            'total_time': q_time,
            'async': True,
            'results': [{'temperature': r['temperature'], 'n': r['n'], 'passed': r['passed'],
                        'pass_rate': r['pass_rate'], 'gen_tps': r['gen_tps'], 'eval_tps': r['eval_tps']}
                       for r in q_results]
        }
        with open(q_dir / f'summaryq{q_idx}.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  Question {q_idx} done in {q_time:.0f}s")

    global_time = time.time() - global_start
    print(f"\n{'='*70}")
    print(f"COMPLETE: {global_time:.0f}s ({global_time/3600:.1f}h)")


if __name__ == "__main__":
    main()
