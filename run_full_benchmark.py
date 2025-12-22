#!/usr/bin/env python3
"""Full benchmark: 175 questions x 16 temperatures x n completions using 4 GPUs in parallel"""
import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path

TEMPERATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 2.0]
NUM_GPUS = 8
GPU_OFFSET = 0  # Use all 8 GPUs
STAGGER_DELAY = 1  # seconds between GPU starts


def run_temperature(question_idx, temp, n_total, output_dir):
    """Run a single temperature across all 4 GPUs in parallel"""
    n_per_gpu = n_total // NUM_GPUS
    remainder = n_total % NUM_GPUS

    # Create output files for each GPU (use absolute paths!)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    temp_str = f"{temp:.1f}".replace('.', '_')
    gpu_files = [f"{output_dir}/gpu{i}_t{temp_str}.json" for i in range(NUM_GPUS)]

    # Launch all 4 GPUs
    processes = []
    for gpu_id in range(NUM_GPUS):
        n_this_gpu = n_per_gpu + (1 if gpu_id < remainder else 0)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id + GPU_OFFSET)  # Skip GPUs 0-1

        cmd = [
            'python3', '/workspace/orkspace/gpu_worker.py',
            str(gpu_id), str(question_idx), str(n_this_gpu), str(temp), gpu_files[gpu_id]
        ]
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((p, gpu_id, gpu_files[gpu_id]))
        time.sleep(STAGGER_DELAY)

    # Wait for all to complete
    results = []
    for p, gpu_id, output_file in processes:
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            print(f"    GPU {gpu_id} error: {stderr.decode()[-200:]}", file=sys.stderr)
        else:
            try:
                with open(output_file) as f:
                    results.append(json.load(f))
                os.remove(output_file)  # Clean up
            except Exception as e:
                print(f"    GPU {gpu_id} failed to load results: {e}", file=sys.stderr)

    return results


def combine_results(gpu_results, question_idx, question_title, temp, n_total):
    """Combine results from multiple GPUs"""
    if not gpu_results:
        return None

    total_tokens = sum(r['total_tokens'] for r in gpu_results)
    max_gen_time = max(r['gen_time'] for r in gpu_results)
    max_eval_time = max(r['eval_time'] for r in gpu_results)
    total_passed = sum(r['passed'] for r in gpu_results)
    total_n = sum(r['n'] for r in gpu_results)

    # Calculate combined TPS metrics
    gen_tps = total_tokens / max_gen_time if max_gen_time > 0 else 0
    eval_tps = total_n / max_eval_time if max_eval_time > 0 else 0
    total_time = max_gen_time + max_eval_time
    pipeline_tps = total_n / total_time if total_time > 0 else 0

    # Combine completions from all GPUs
    all_completions = []
    for r in gpu_results:
        all_completions.extend(r.get('completions', []))

    # Get reproducibility params from first GPU result
    first = gpu_results[0]

    # Get earliest start and latest end across all GPUs
    start_times = [r.get('start_time') for r in gpu_results if r.get('start_time')]
    end_times = [r.get('end_time') for r in gpu_results if r.get('end_time')]

    return {
        # Timestamps
        'start_time': min(start_times) if start_times else None,
        'end_time': max(end_times) if end_times else None,
        # Reproducibility params
        'model': first.get('model', 'Qwen/Qwen3-0.6B'),
        'prompt': first.get('prompt', ''),
        'sampling_params': first.get('sampling_params', {}),
        'model_config': first.get('model_config', {}),
        # Question info
        'question_idx': question_idx,
        'question_title': question_title,
        'temperature': temp,
        # Run stats
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
        'completions': all_completions,  # List of {"raw": ..., "passed": true/false}
        'gpu_results': [{'gpu_id': r['gpu_id'], 'n': r['n'], 'gen_tps': r['gen_tps'], 'eval_tps': r.get('eval_tps', 0), 'passed': r['passed']} for r in gpu_results]
    }


def main():
    parser = argparse.ArgumentParser(description="Full benchmark with 4-GPU parallelization")
    parser.add_argument('--start', type=int, default=0, help='Start question index')
    parser.add_argument('--end', type=int, default=175, help='End question index (exclusive)')
    parser.add_argument('--n', type=int, default=10000, help='Total completions per temperature')
    parser.add_argument('--output_dir', '-o', type=str, default='results_full', help='Output directory')
    parser.add_argument('--temps', type=str, default=None, help='Comma-separated temperatures (default: all 16)')
    args = parser.parse_args()

    # Parse temperatures
    if args.temps:
        temps = [float(t) for t in args.temps.split(',')]
    else:
        temps = TEMPERATURES

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset to get question titles
    sys.path.insert(0, '/workspace/orkspace/LiveCodeBench')
    os.chdir('/workspace/orkspace/LiveCodeBench')
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    dataset = load_code_generation_dataset(release_version="v6")
    os.chdir('/workspace/orkspace')

    print(f"{'='*70}")
    print(f"FULL BENCHMARK: {args.end - args.start} questions x {len(temps)} temps x {args.n} completions")
    print(f"Using {NUM_GPUS} GPUs in parallel ({args.n // NUM_GPUS} per GPU)")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    global_start = time.time()
    total_questions = 0
    total_temps_run = 0

    for q_idx in range(args.start, min(args.end, len(dataset))):
        problem = dataset[q_idx]
        q_dir = output_dir / f"q{q_idx}"
        q_dir.mkdir(exist_ok=True)

        print(f"\n[Q{q_idx}] {problem.question_title}")
        q_start = time.time()
        q_results = []

        for temp in temps:
            print(f"  T={temp:.1f}: ", end='', flush=True)
            t_start = time.time()

            gpu_results = run_temperature(q_idx, temp, args.n, str(q_dir))
            combined = combine_results(gpu_results, q_idx, problem.question_title, temp, args.n)

            if combined:
                # Save temperature result
                temp_str = f"{temp:.1f}".replace('.', '_')
                with open(q_dir / f"t{temp_str}_n{args.n}.json", 'w') as f:
                    json.dump(combined, f, indent=2)

                q_results.append(combined)
                t_time = time.time() - t_start
                print(f"gen:{combined['gen_tps']:,.0f} eval:{combined['eval_tps']:,.0f} pipe:{combined['pipeline_tps']:,.0f} | {combined['passed']}/{combined['n']} ({combined['pass_rate']:.1f}%) [{t_time:.0f}s]")
                total_temps_run += 1
            else:
                print("FAILED")

        # Save question summary
        q_time = time.time() - q_start
        summary = {
            'question_idx': q_idx,
            'question_title': problem.question_title,
            'n': args.n,
            'temperatures': temps,
            'total_time': q_time,
            'results': [{'temperature': r['temperature'], 'n': r['n'], 'passed': r['passed'], 'pass_rate': r['pass_rate'], 'gen_tps': r['gen_tps'], 'eval_tps': r['eval_tps'], 'pipeline_tps': r['pipeline_tps']} for r in q_results]
        }
        with open(q_dir / f'summaryq{q_idx}.json', 'w') as f:
            json.dump(summary, f, indent=2)

        total_questions += 1
        print(f"  Question {q_idx} done in {q_time:.0f}s")

    # Global summary
    global_time = time.time() - global_start
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Questions: {total_questions}")
    print(f"Temps per question: {len(temps)}")
    print(f"Total temp runs: {total_temps_run}")
    print(f"Total time: {global_time:.0f}s ({global_time/3600:.1f}h)")

    # Save global summary
    with open(output_dir / 'run_summary.json', 'w') as f:
        json.dump({
            'start_idx': args.start,
            'end_idx': args.end,
            'n': args.n,
            'temperatures': temps,
            'num_gpus': NUM_GPUS,
            'total_questions': total_questions,
            'total_temps_run': total_temps_run,
            'total_time': global_time,
        }, f, indent=2)


if __name__ == "__main__":
    main()
