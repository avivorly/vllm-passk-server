#!/usr/bin/env python3
"""Optimal benchmark: Each GPU loads model ONCE and processes all assigned questions"""
import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path

TEMPERATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 2.0]
NUM_GPUS = 8
GPU_OFFSET = 0


def main():
    parser = argparse.ArgumentParser(description="Optimal benchmark - ONE model load per GPU for entire run")
    parser.add_argument('--start', type=int, default=0, help='Start question index')
    parser.add_argument('--end', type=int, default=175, help='End question index')
    parser.add_argument('--n', type=int, default=3000, help='Total completions per temperature')
    parser.add_argument('--output_dir', '-o', type=str, default='results_optimal', help='Output directory')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total_questions = args.end - args.start
    n_per_gpu = args.n // NUM_GPUS
    temps_str = ','.join(str(t) for t in TEMPERATURES)

    print(f"{'='*70}")
    print(f"OPTIMAL BENCHMARK")
    print(f"{'='*70}")
    print(f"Questions: {args.start} to {args.end} ({total_questions} total)")
    print(f"Temperatures: {len(TEMPERATURES)}")
    print(f"Completions: {args.n} total ({n_per_gpu} per GPU)")
    print(f"GPUs: {NUM_GPUS}")
    print(f"Model loads: {NUM_GPUS} (ONE per GPU for entire run!)")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    global_start = time.time()

    # Launch all 8 GPUs - each processes ALL questions with ONE model load
    processes = []
    for gpu_id in range(NUM_GPUS):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id + GPU_OFFSET)

        cmd = [
            'python3', '/workspace/orkspace/gpu_worker_full.py',
            str(gpu_id),
            str(args.start),
            str(args.end),
            str(n_per_gpu),
            output_dir,
            temps_str
        ]

        log_file = open(f"{output_dir}/gpu{gpu_id}.log", 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((p, gpu_id, log_file))
        print(f"[GPU {gpu_id}] Started worker for questions {args.start}-{args.end}")

    print(f"\nAll {NUM_GPUS} GPUs launched! Each loading model ONCE.")
    print(f"Monitor progress: tail -f {output_dir}/gpu*.log")
    print(f"GPU status: nvidia-smi")

    # Wait for all to complete
    for p, gpu_id, log_file in processes:
        p.wait()
        log_file.close()
        if p.returncode == 0:
            print(f"[GPU {gpu_id}] Completed successfully")
        else:
            print(f"[GPU {gpu_id}] Failed with code {p.returncode}")

    global_time = time.time() - global_start

    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {global_time:.0f}s ({global_time/3600:.2f}h)")
    print(f"Questions: {total_questions}")
    print(f"Temps per question: {len(TEMPERATURES)}")
    print(f"Total runs: {total_questions * len(TEMPERATURES)}")

    # Save summary
    with open(f"{output_dir}/run_summary.json", 'w') as f:
        json.dump({
            'start_idx': args.start,
            'end_idx': args.end,
            'n': args.n,
            'n_per_gpu': n_per_gpu,
            'temperatures': TEMPERATURES,
            'num_gpus': NUM_GPUS,
            'total_time_seconds': global_time,
        }, f, indent=2)


if __name__ == "__main__":
    main()
