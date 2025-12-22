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


def combine_gpu_results(output_dir, start_q, end_q, n_total):
    """Combine results from all GPUs into single files per temperature"""
    output_dir = Path(output_dir)

    for q_idx in range(start_q, end_q):
        q_dir = output_dir / f"q{q_idx}"
        if not q_dir.exists():
            continue

        for temp in TEMPERATURES:
            temp_str = f"{temp:.1f}".replace('.', '_')

            # Collect results from all GPUs
            gpu_results = []
            for gpu_id in range(NUM_GPUS):
                gpu_file = q_dir / f"gpu{gpu_id}_t{temp_str}.json"
                if gpu_file.exists():
                    with open(gpu_file) as f:
                        gpu_results.append(json.load(f))

            if not gpu_results:
                continue

            # Combine results
            total_tokens = sum(r['total_tokens'] for r in gpu_results)
            max_gen_time = max(r['gen_time'] for r in gpu_results)
            max_eval_time = max(r['eval_time'] for r in gpu_results)
            total_passed = sum(r['passed'] for r in gpu_results)
            total_n = sum(r['n'] for r in gpu_results)

            gen_tps = total_tokens / max_gen_time if max_gen_time > 0 else 0
            eval_tps = total_n / max_eval_time if max_eval_time > 0 else 0
            total_time = max_gen_time + max_eval_time

            # Combine completions
            all_completions = []
            for r in gpu_results:
                all_completions.extend(r.get('completions', []))

            first = gpu_results[0]
            combined = {
                'timing': first.get('timing', {}),
                'model': first.get('model', 'Qwen/Qwen3-0.6B'),
                'prompt': first.get('prompt', ''),
                'sampling_params': first.get('sampling_params', {}),
                'model_config': first.get('model_config', {}),
                'question_idx': q_idx,
                'question_title': first.get('question_title', ''),
                'temperature': temp,
                'n': total_n,
                'total_tokens': total_tokens,
                'gen_time': max_gen_time,
                'gen_tps': gen_tps,
                'eval_time': max_eval_time,
                'eval_tps': eval_tps,
                'total_time': total_time,
                'passed': total_passed,
                'pass_rate': total_passed / total_n * 100 if total_n > 0 else 0,
                'completions': all_completions,
                'gpu_breakdown': [{'gpu_id': r['gpu_id'], 'n': r['n'], 'passed': r['passed'], 'gen_tps': r['gen_tps']} for r in gpu_results],
            }

            # Save combined result
            combined_file = q_dir / f"t{temp_str}_n{n_total}.json"
            with open(combined_file, 'w') as f:
                json.dump(combined, f, indent=2)

        # Create question summary
        q_results = []
        for temp in TEMPERATURES:
            temp_str = f"{temp:.1f}".replace('.', '_')
            combined_file = q_dir / f"t{temp_str}_n{n_total}.json"
            if combined_file.exists():
                with open(combined_file) as f:
                    data = json.load(f)
                    q_results.append({
                        'temperature': temp,
                        'n': data['n'],
                        'passed': data['passed'],
                        'pass_rate': data['pass_rate'],
                        'gen_tps': data['gen_tps'],
                    })

        if q_results:
            with open(q_dir / f"summary_q{q_idx}.json", 'w') as f:
                json.dump({
                    'question_idx': q_idx,
                    'n': n_total,
                    'results': q_results,
                }, f, indent=2)

    print(f"Combined results for {end_q - start_q} questions")


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
