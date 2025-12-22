#!/usr/bin/env python3
"""Combiner worker: Watches for raw GPU files, combines them, runs evaluation"""
import os
import sys
import json
import time
import re
import glob
import datetime
from pathlib import Path

# Args: output_dir n_total num_gpus
output_dir = Path(os.path.abspath(sys.argv[1]))
n_total = int(sys.argv[2])
num_gpus = int(sys.argv[3])
worker_id = int(sys.argv[4]) if len(sys.argv) > 4 else 0

TEMPERATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 2.0]

# Setup LiveCodeBench for evaluation
os.chdir('/workspace/orkspace/LiveCodeBench')
sys.path.insert(0, '/workspace/orkspace/LiveCodeBench')
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

def extract_code(text):
    pattern = r'```python\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    pattern = r'```\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()

print(f"[Combiner {worker_id}] Loading dataset for evaluation...", file=sys.stderr)
dataset = load_code_generation_dataset(release_version="v6")

def process_question_temp(q_idx, temp):
    """Combine GPU files and run evaluation for one question+temp"""
    q_dir = output_dir / f"q{q_idx}"
    temp_str = f"{temp:.1f}".replace('.', '_')

    # Check if already processed
    combined_file = q_dir / f"t{temp_str}_n{n_total}.json"
    if combined_file.exists():
        return None  # Already done

    # Check if all GPU raw files exist
    gpu_files = []
    for gpu_id in range(num_gpus):
        raw_file = q_dir / f"gpu{gpu_id}_t{temp_str}_raw.json"
        if not raw_file.exists():
            return None  # Not all GPUs done yet
        gpu_files.append(raw_file)

    # Load all GPU results
    gpu_results = []
    all_raw_outputs = []
    for raw_file in gpu_files:
        with open(raw_file) as f:
            data = json.load(f)
            gpu_results.append(data)
            all_raw_outputs.extend(data['raw_outputs'])

    # Extract code from raw outputs
    completions = [extract_code(raw) for raw in all_raw_outputs]

    # Run evaluation
    problem = dataset[q_idx]
    eval_sample = problem.get_evaluation_sample()

    eval_start = time.time()
    eval_start_time = datetime.datetime.now().isoformat()

    metrics, results, _ = codegen_metrics(
        samples_list=[eval_sample],
        generations_list=[completions],
        k_list=[1],
        num_process_evaluate=24,
        timeout=6,
        debug=False,
    )

    eval_time = time.time() - eval_start
    eval_end_time = datetime.datetime.now().isoformat()

    # Count passed
    pass_list = [all(x == True for x in r) for r in results[0]]
    passed_count = sum(pass_list)

    # Combine stats
    total_tokens = sum(r['total_tokens'] for r in gpu_results)
    max_gen_time = max(r['gen_time'] for r in gpu_results)
    total_n = sum(r['n'] for r in gpu_results)

    gen_tps = total_tokens / max_gen_time if max_gen_time > 0 else 0
    eval_tps = total_n / eval_time if eval_time > 0 else 0

    # Build combined result
    first = gpu_results[0]
    combined = {
        'status': 'evaluated',
        'model': 'Qwen/Qwen3-0.6B',
        'prompt': first['prompt'],
        'sampling_params': {
            'temperature': temp,
            'max_tokens': 500,
            'stop': ["```\n```", "```\n\n", "\n\n\n"],
        },
        'question_idx': q_idx,
        'question_title': first['question_title'],
        'temperature': temp,
        'n': total_n,
        'total_tokens': total_tokens,
        'gen_time': max_gen_time,
        'gen_tps': gen_tps,
        'eval_time': eval_time,
        'eval_tps': eval_tps,
        'passed': passed_count,
        'pass_rate': passed_count / total_n * 100 if total_n > 0 else 0,
        'timing': {
            'generation': {
                'start_time': min(r['timing']['gen_start'] for r in gpu_results),
                'end_time': max(r['timing']['gen_end'] for r in gpu_results),
                'duration_seconds': max_gen_time,
            },
            'evaluation': {
                'start_time': eval_start_time,
                'end_time': eval_end_time,
                'duration_seconds': eval_time,
            },
        },
        'completions': [
            {'raw': raw, 'code': code, 'passed': passed}
            for raw, code, passed in zip(all_raw_outputs, completions, pass_list)
        ],
        'gpu_breakdown': [
            {'gpu_id': r['gpu_id'], 'n': r['n'], 'gen_tps': r['gen_tps']}
            for r in gpu_results
        ],
    }

    # Save combined result
    with open(combined_file, 'w') as f:
        json.dump(combined, f, indent=2)

    # Delete raw files to save space
    for raw_file in gpu_files:
        os.remove(raw_file)

    return {
        'q_idx': q_idx,
        'temp': temp,
        'passed': passed_count,
        'total': total_n,
        'pass_rate': combined['pass_rate'],
    }

# Main loop: continuously scan for work
print(f"[Combiner {worker_id}] Starting - scanning for raw files...", file=sys.stderr)
processed = 0
scan_count = 0

while True:
    found_work = False

    # Scan all question directories
    for q_idx in range(175):
        for temp in TEMPERATURES:
            result = process_question_temp(q_idx, temp)
            if result:
                processed += 1
                found_work = True
                print(f"[Combiner {worker_id}] Q{result['q_idx']} T={result['temp']}: {result['passed']}/{result['total']} ({result['pass_rate']:.1f}%)", file=sys.stderr)

    scan_count += 1

    # Check if GPUs are still running
    gpu_logs_exist = any((output_dir / f"gpu{i}.log").exists() for i in range(num_gpus))

    if not found_work:
        if scan_count > 5 and not gpu_logs_exist:
            # No more work and GPUs are done
            break
        time.sleep(2)  # Wait before next scan

print(f"[Combiner {worker_id}] DONE - processed {processed} question/temp combinations", file=sys.stderr)
