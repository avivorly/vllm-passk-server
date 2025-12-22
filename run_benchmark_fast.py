#!/usr/bin/env python3
"""Fast benchmark: Load model ONCE per GPU, run all temps without reloading"""
import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path

TEMPERATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 2.0]
NUM_GPUS = 8
GPU_OFFSET = 0


def run_question(question_idx, n_per_gpu, output_dir, temps):
    """Run ALL temperatures for one question across all GPUs (model loaded ONCE per GPU)"""

    # Use absolute paths to avoid issues with worker cwd changes
    output_dir = os.path.abspath(output_dir)
    q_dir = f"{output_dir}/q{question_idx}"
    os.makedirs(q_dir, exist_ok=True)

    temps_str = ','.join(str(t) for t in temps)

    # Launch all GPUs - each processes ALL temps with ONE model load
    processes = []
    for gpu_id in range(NUM_GPUS):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id + GPU_OFFSET)

        cmd = [
            'python3', '/workspace/orkspace/gpu_worker_multi_temp.py',
            str(gpu_id), str(question_idx), str(n_per_gpu), q_dir, temps_str
        ]
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((p, gpu_id))

    # Wait for all to complete
    results_by_temp = {t: [] for t in temps}
    for p, gpu_id in processes:
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            print(f"  GPU {gpu_id} error: {stderr.decode()[-500:]}", file=sys.stderr)
        else:
            # Load results for each temp from this GPU
            for temp in temps:
                temp_str = f"{temp:.1f}".replace('.', '_')
                result_file = f"{q_dir}/gpu{gpu_id}_t{temp_str}.json"
                try:
                    with open(result_file) as f:
                        results_by_temp[temp].append(json.load(f))
                except Exception as e:
                    print(f"  GPU {gpu_id} T={temp} failed: {e}", file=sys.stderr)

    return results_by_temp


def combine_gpu_results(gpu_results, question_idx, question_title, temp, n_total):
    """Combine results from multiple GPUs for one temperature"""
    if not gpu_results:
        return None

    total_tokens = sum(r['total_tokens'] for r in gpu_results)
    max_gen_time = max(r['gen_time'] for r in gpu_results)
    max_eval_time = max(r['eval_time'] for r in gpu_results)
    total_passed = sum(r['passed'] for r in gpu_results)
    total_n = sum(r['n'] for r in gpu_results)

    gen_tps = total_tokens / max_gen_time if max_gen_time > 0 else 0
    eval_tps = total_n / max_eval_time if max_eval_time > 0 else 0
    total_time = max_gen_time + max_eval_time
    pipeline_tps = total_n / total_time if total_time > 0 else 0

    all_completions = []
    for r in gpu_results:
        all_completions.extend(r.get('completions', []))

    first = gpu_results[0]

    return {
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
    }


def main():
    parser = argparse.ArgumentParser(description="Fast benchmark - one model load per GPU")
    parser.add_argument('--start', type=int, default=0, help='Start question index')
    parser.add_argument('--end', type=int, default=175, help='End question index')
    parser.add_argument('--n', type=int, default=3000, help='Total completions per temperature')
    parser.add_argument('--output_dir', '-o', type=str, default='results_fast', help='Output directory')
    args = parser.parse_args()

    n_per_gpu = args.n // NUM_GPUS

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    sys.path.insert(0, '/workspace/orkspace/LiveCodeBench')
    os.chdir('/workspace/orkspace/LiveCodeBench')
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    dataset = load_code_generation_dataset(release_version="v6")
    os.chdir('/workspace/orkspace')

    print(f"{'='*70}")
    print(f"FAST BENCHMARK: {args.end - args.start} questions x {len(TEMPERATURES)} temps x {args.n} completions")
    print(f"Using {NUM_GPUS} GPUs - model loaded ONCE per GPU per question")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    global_start = time.time()

    for q_idx in range(args.start, min(args.end, len(dataset))):
        problem = dataset[q_idx]
        q_dir = output_dir / f"q{q_idx}"

        print(f"\n[Q{q_idx}] {problem.question_title}")
        q_start = time.time()

        # Run all temps for this question (model loaded once per GPU)
        results_by_temp = run_question(q_idx, n_per_gpu, str(output_dir), TEMPERATURES)

        # Combine and save results for each temp
        q_results = []
        for temp in TEMPERATURES:
            gpu_results = results_by_temp.get(temp, [])
            combined = combine_gpu_results(gpu_results, q_idx, problem.question_title, temp, args.n)

            if combined:
                temp_str = f"{temp:.1f}".replace('.', '_')
                with open(q_dir / f"t{temp_str}_n{args.n}.json", 'w') as f:
                    json.dump(combined, f, indent=2)

                print(f"  T={temp:.1f}: gen:{combined['gen_tps']:,.0f} eval:{combined['eval_tps']:,.0f} | {combined['passed']}/{combined['n']} ({combined['pass_rate']:.1f}%)")
                q_results.append({
                    'temperature': temp,
                    'n': combined['n'],
                    'passed': combined['passed'],
                    'pass_rate': combined['pass_rate'],
                    'gen_tps': combined['gen_tps'],
                })

        # Save question summary
        q_time = time.time() - q_start
        with open(q_dir / f'summary_q{q_idx}.json', 'w') as f:
            json.dump({
                'question_idx': q_idx,
                'question_title': problem.question_title,
                'n': args.n,
                'total_time': q_time,
                'results': q_results,
            }, f, indent=2)

        print(f"  Question {q_idx} done in {q_time:.0f}s")

    global_time = time.time() - global_start
    print(f"\n{'='*70}")
    print(f"COMPLETE: {global_time:.0f}s ({global_time/3600:.1f}h)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
