#!/usr/bin/env python3
"""Pipeline benchmark: GPU workers generate, Combiner workers evaluate in parallel"""
import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path

TEMPERATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 2.0]
NUM_GPUS = 8
NUM_COMBINERS = 24  # CPU workers for evaluation


def main():
    parser = argparse.ArgumentParser(description="Pipeline benchmark - GPU generates, CPU evaluates in parallel")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=175)
    parser.add_argument('--n', type=int, default=3000, help='Total completions per temp')
    parser.add_argument('--output_dir', '-o', type=str, default='results_pipeline')
    parser.add_argument('--combiners', type=int, default=NUM_COMBINERS)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    n_per_gpu = args.n // NUM_GPUS
    temps_str = ','.join(str(t) for t in TEMPERATURES)

    print(f"{'='*70}")
    print(f"PIPELINE BENCHMARK")
    print(f"{'='*70}")
    print(f"Questions: {args.start}-{args.end}")
    print(f"Completions: {args.n} ({n_per_gpu} per GPU)")
    print(f"GPU workers: {NUM_GPUS} (generation only)")
    print(f"Combiner workers: {args.combiners} (evaluation)")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    global_start = time.time()

    # Launch GPU workers (generation only) with staggered starts
    gpu_processes = []
    for gpu_id in range(NUM_GPUS):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        cmd = [
            'python3', '/workspace/orkspace/gpu_worker_gen_only.py',
            str(gpu_id), str(args.start), str(args.end),
            str(n_per_gpu), output_dir, temps_str
        ]

        log_file = open(f"{output_dir}/gpu{gpu_id}.log", 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        gpu_processes.append((p, gpu_id, log_file))
        print(f"[GPU {gpu_id}] Started generation worker")
        time.sleep(2)  # Stagger GPU starts to avoid init conflicts

    # Launch combiner workers (evaluation)
    combiner_processes = []
    for c_id in range(args.combiners):
        cmd = [
            'python3', '/workspace/orkspace/combiner_worker.py',
            output_dir, str(args.n), str(NUM_GPUS), str(c_id)
        ]

        log_file = open(f"{output_dir}/combiner{c_id}.log", 'w')
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        combiner_processes.append((p, c_id, log_file))

    print(f"[Combiners] Started {args.combiners} evaluation workers")
    print(f"\nMonitor: tail -f {output_dir}/gpu*.log {output_dir}/combiner*.log")

    # Wait for GPU workers to finish
    print("\nWaiting for GPU workers...")
    for p, gpu_id, log_file in gpu_processes:
        p.wait()
        log_file.close()
        print(f"[GPU {gpu_id}] Finished")

    # Signal combiners that GPUs are done by removing gpu logs
    print("\nGPUs done! Waiting for combiners to finish remaining work...")

    # Wait for combiners (they'll exit when no more work)
    for p, c_id, log_file in combiner_processes:
        p.wait()
        log_file.close()

    global_time = time.time() - global_start

    # Generate final summaries
    print("\nGenerating summaries...")
    for q_idx in range(args.start, args.end):
        q_dir = Path(output_dir) / f"q{q_idx}"
        if not q_dir.exists():
            continue

        q_results = []
        for temp in TEMPERATURES:
            temp_str = f"{temp:.1f}".replace('.', '_')
            combined_file = q_dir / f"t{temp_str}_n{args.n}.json"
            if combined_file.exists():
                with open(combined_file) as f:
                    data = json.load(f)
                    q_results.append({
                        'temperature': temp,
                        'n': data['n'],
                        'passed': data['passed'],
                        'pass_rate': data['pass_rate'],
                        'gen_tps': data['gen_tps'],
                        'eval_tps': data['eval_tps'],
                    })

        if q_results:
            with open(q_dir / f"summary_q{q_idx}.json", 'w') as f:
                json.dump({
                    'question_idx': q_idx,
                    'n': args.n,
                    'results': q_results,
                }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {global_time:.0f}s ({global_time/3600:.2f}h)")

    with open(f"{output_dir}/run_summary.json", 'w') as f:
        json.dump({
            'start': args.start,
            'end': args.end,
            'n': args.n,
            'num_gpus': NUM_GPUS,
            'num_combiners': args.combiners,
            'total_time': global_time,
        }, f, indent=2)


if __name__ == "__main__":
    main()
