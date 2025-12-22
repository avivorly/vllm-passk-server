#!/usr/bin/env python3
"""Pipelined benchmark: GPUs grab next temp immediately when free (no waiting)"""
import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from queue import Queue
from threading import Thread, Lock

TEMPERATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 2.0]
NUM_GPUS = 8
N_PER_TEMP = 3000  # Total n per temperature (distributed across GPUs that grab it)


def gpu_worker_thread(gpu_id, job_queue, results_dict, results_lock, output_dir):
    """Worker thread for one GPU - grabs jobs from queue until empty"""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['HF_HOME'] = '/workspace/.cache/huggingface'

    while True:
        try:
            job = job_queue.get_nowait()
        except:
            break  # Queue empty, worker done

        question_idx, temp, n_for_this_job = job
        temp_str = f"{temp:.1f}".replace('.', '_')
        output_file = f"{output_dir}/gpu{gpu_id}_t{temp_str}.json"

        cmd = [
            'python3', '/workspace/orkspace/gpu_worker.py',
            str(gpu_id), str(question_idx), str(n_for_this_job), str(temp), output_file
        ]

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(output_file):
            try:
                with open(output_file) as f:
                    data = json.load(f)
                with results_lock:
                    if temp not in results_dict:
                        results_dict[temp] = []
                    results_dict[temp].append(data)
            except Exception as e:
                print(f"  [GPU{gpu_id}] Failed to load T={temp}: {e}", file=sys.stderr)
        else:
            print(f"  [GPU{gpu_id}] T={temp} failed: {result.stderr[-200:]}", file=sys.stderr)

        job_queue.task_done()


def run_question_pipelined(question_idx, question_title, temps, n_total, output_dir):
    """Run all temperatures for one question with pipelined GPU utilization"""

    # Create job queue: each temp gets split across multiple GPU jobs
    # We'll have more jobs than GPUs so they stay busy
    job_queue = Queue()

    # Split each temperature into smaller chunks so GPUs can grab work faster
    jobs_per_temp = 2  # Split each temp into 2 jobs (helps pipelining)
    n_per_job = n_total // jobs_per_temp

    for temp in temps:
        for _ in range(jobs_per_temp):
            job_queue.put((question_idx, temp, n_per_job))

    # Results storage
    results_dict = {}  # temp -> list of GPU results
    results_lock = Lock()

    # Start worker threads (one per GPU)
    threads = []
    for gpu_id in range(NUM_GPUS):
        t = Thread(target=gpu_worker_thread,
                   args=(gpu_id, job_queue, results_dict, results_lock, output_dir))
        t.start()
        threads.append(t)
        time.sleep(1)  # Slight stagger to avoid init conflicts

    # Wait for all work to complete
    for t in threads:
        t.join()

    return results_dict


def combine_temp_results(gpu_results_list, question_idx, question_title, temp, n_total):
    """Combine results from multiple GPU runs for same temperature"""
    if not gpu_results_list:
        return None

    total_tokens = sum(r['total_tokens'] for r in gpu_results_list)
    max_gen_time = max(r['gen_time'] for r in gpu_results_list)
    max_eval_time = max(r['eval_time'] for r in gpu_results_list)
    total_passed = sum(r['passed'] for r in gpu_results_list)
    total_n = sum(r['n'] for r in gpu_results_list)

    gen_tps = total_tokens / max_gen_time if max_gen_time > 0 else 0
    eval_tps = total_n / max_eval_time if max_eval_time > 0 else 0
    total_time = max_gen_time + max_eval_time
    pipeline_tps = total_n / total_time if total_time > 0 else 0

    # Combine completions
    all_completions = []
    for r in gpu_results_list:
        all_completions.extend(r.get('completions', []))

    first = gpu_results_list[0]
    start_times = [r.get('start_time') for r in gpu_results_list if r.get('start_time')]
    end_times = [r.get('end_time') for r in gpu_results_list if r.get('end_time')]

    return {
        'start_time': min(start_times) if start_times else None,
        'end_time': max(end_times) if end_times else None,
        'model': first.get('model', 'Qwen/Qwen3-0.6B'),
        'prompt': first.get('prompt', ''),
        'sampling_params': first.get('sampling_params', {}),
        'model_config': first.get('model_config', {}),
        'question_idx': question_idx,
        'question_title': question_title,
        'temperature': temp,
        'n': total_n,
        'total_tokens': total_tokens,
        'gen_time': max_gen_time,
        'gen_tps': gen_tps,
        'eval_time': max_eval_time,
        'eval_tps': eval_tps,
        'total_time': total_time,
        'pipeline_tps': pipeline_tps,
        'passed': total_passed,
        'pass_rate': total_passed / total_n * 100 if total_n > 0 else 0,
        'completions': all_completions,
        'gpu_results': [{'gpu_id': r['gpu_id'], 'n': r['n'], 'gen_tps': r['gen_tps'], 'passed': r['passed']} for r in gpu_results_list]
    }


def main():
    parser = argparse.ArgumentParser(description="Pipelined benchmark - GPUs always busy")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=175)
    parser.add_argument('--n', type=int, default=3000)
    parser.add_argument('--output_dir', '-o', type=str, default='results_pipelined')
    parser.add_argument('--temps', type=str, default=None)
    args = parser.parse_args()

    temps = [float(t) for t in args.temps.split(',')] if args.temps else TEMPERATURES

    output_dir = Path(args.output_dir).resolve()  # Make absolute
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    sys.path.insert(0, '/workspace/orkspace/LiveCodeBench')
    os.chdir('/workspace/orkspace/LiveCodeBench')
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    dataset = load_code_generation_dataset(release_version="v6")
    os.chdir('/workspace/orkspace')

    print(f"{'='*70}")
    print(f"PIPELINED BENCHMARK: {args.end - args.start} questions x {len(temps)} temps x {args.n} completions")
    print(f"Using {NUM_GPUS} GPUs with pipelined execution (no waiting)")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    global_start = time.time()

    for q_idx in range(args.start, min(args.end, len(dataset))):
        problem = dataset[q_idx]
        q_dir = output_dir / f"q{q_idx}"
        q_dir.mkdir(exist_ok=True)

        print(f"\n[Q{q_idx}] {problem.question_title}")
        q_start = time.time()

        # Run all temps with pipelining
        results_dict = run_question_pipelined(q_idx, problem.question_title, temps, args.n, str(q_dir))

        # Process and save results for each temp
        q_results = []
        for temp in temps:
            gpu_results = results_dict.get(temp, [])
            combined = combine_temp_results(gpu_results, q_idx, problem.question_title, temp, args.n)

            if combined:
                temp_str = f"{temp:.1f}".replace('.', '_')
                with open(q_dir / f"t{temp_str}_n{args.n}.json", 'w') as f:
                    json.dump(combined, f, indent=2)

                q_results.append(combined)
                print(f"  T={temp:.1f}: gen:{combined['gen_tps']:,.0f} eval:{combined['eval_tps']:,.0f} | {combined['passed']}/{combined['n']} ({combined['pass_rate']:.1f}%)")

                # Clean up temp GPU files
                for gpu_id in range(NUM_GPUS):
                    gpu_file = q_dir / f"gpu{gpu_id}_t{temp_str}.json"
                    if gpu_file.exists():
                        gpu_file.unlink()

        q_time = time.time() - q_start

        # Save summary
        summary = {
            'question_idx': q_idx,
            'question_title': problem.question_title,
            'n': args.n,
            'temperatures': temps,
            'total_time': q_time,
            'pipelined': True,
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
    print(f"{'='*70}")

    with open(output_dir / 'run_summary.json', 'w') as f:
        json.dump({
            'start_idx': args.start,
            'end_idx': args.end,
            'n': args.n,
            'temperatures': temps,
            'num_gpus': NUM_GPUS,
            'pipelined': True,
            'total_time': global_time,
        }, f, indent=2)


if __name__ == "__main__":
    main()
