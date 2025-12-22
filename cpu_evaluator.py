#!/usr/bin/env python3
"""CPU Evaluator: Watches pending queue, evaluates completions, writes final results.
   Multiple evaluators can run in parallel for high CPU utilization."""
import os
import sys
import json
import time
import glob
import signal
import datetime

# Get args: evaluator_id output_dir num_workers
evaluator_id = int(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])
num_eval_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 24

pending_eval_dir = f"{output_dir}/pending_eval"
pending_combine_dir = f"{output_dir}/pending_combine"
os.makedirs(pending_combine_dir, exist_ok=True)

# Setup LiveCodeBench
os.chdir('/workspace/orkspace/LiveCodeBench')
sys.path.insert(0, '/workspace/orkspace/LiveCodeBench')

from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

# Control flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    global running
    print(f"[EVAL {evaluator_id}] Received signal {signum}, shutting down...", file=sys.stderr)
    running = False

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

print(f"[EVAL {evaluator_id}] Starting CPU evaluator (workers={num_eval_workers})", file=sys.stderr)
print(f"[EVAL {evaluator_id}] Watching: {pending_eval_dir}", file=sys.stderr)
print(f"[EVAL {evaluator_id}] Output to: {pending_combine_dir}", file=sys.stderr)

# Load dataset for evaluation samples
dataset = load_code_generation_dataset(release_version="v6")
print(f"[EVAL {evaluator_id}] Dataset loaded ({len(dataset)} problems)", file=sys.stderr)

processed_count = 0

while running:
    # Find pending files (only base files, not already claimed)
    pending_files = [f for f in glob.glob(f"{pending_eval_dir}/q*_t*_gpu*.json")
                     if '.processing' not in f]

    if not pending_files:
        # No work, sleep briefly
        time.sleep(0.5)
        continue

    # Process first available file
    for pending_file in pending_files:
        if not running:
            break

        # Try to claim this file by renaming it
        processing_file = pending_file.replace(".json", f".processing{evaluator_id}.json")
        try:
            os.rename(pending_file, processing_file)
        except FileNotFoundError:
            # Another evaluator got it first
            continue
        except Exception as e:
            print(f"[EVAL {evaluator_id}] Error claiming {pending_file}: {e}", file=sys.stderr)
            continue

        # Process this file
        try:
            with open(processing_file) as f:
                data = json.load(f)

            question_idx = data["question_idx"]
            temp = data["temperature"]
            gpu_id = data["gpu_id"]
            completions = data["completions"]
            n = data["n"]

            # Get evaluation sample
            problem = dataset[question_idx]
            eval_sample = problem.get_evaluation_sample()

            eval_start_time = datetime.datetime.now().isoformat()
            eval_start = time.time()

            # Run evaluation
            metrics, results, _ = codegen_metrics(
                samples_list=[eval_sample],
                generations_list=[completions],
                k_list=[1],
                num_process_evaluate=num_eval_workers,
                timeout=6,
                debug=False,
            )

            eval_end = time.time()
            eval_time = eval_end - eval_start
            eval_tps = n / eval_time

            passed_count = sum(1 for r in results[0] if all(x == True for x in r))
            pass_list = [all(x == True for x in r) for r in results[0]]

            # Build final result
            completions_with_status = [
                {"raw": raw, "passed": passed}
                for raw, passed in zip(data["raw_outputs"], pass_list)
            ]

            total_time = data["gen_time"] + eval_time

            result = {
                "timing": {
                    "generation": {
                        "start_time": data["gen_start_time"],
                        "end_time": data["gen_end_time"],
                        "duration_seconds": round(data["gen_time"], 3),
                    },
                    "evaluation": {
                        "start_time": eval_start_time,
                        "end_time": datetime.datetime.now().isoformat(),
                        "duration_seconds": round(eval_time, 3),
                        "evaluator_id": evaluator_id,
                    },
                    "overall": {
                        "duration_seconds": round(total_time, 3),
                    },
                    "model_load_seconds": round(data["model_load_time"], 3),
                },
                "model": "Qwen/Qwen3-0.6B",
                "prompt": data["prompt"],
                "sampling_params": {
                    "temperature": temp,
                    "max_tokens": 500,
                    "stop": ["```\n```", "```\n\n", "\n\n\n"],
                },
                "model_config": {
                    "max_model_len": 2182,
                    "gpu_memory_utilization": 0.95,
                    "dtype": "half",
                    "max_num_seqs": 512,
                    "max_num_batched_tokens": 32768,
                },
                "question_idx": question_idx,
                "question_title": data["question_title"],
                "gpu_id": gpu_id,
                "temperature": temp,
                "n": n,
                "total_tokens": data["total_tokens"],
                "gen_time": data["gen_time"],
                "gen_tps": data["gen_tps"],
                "eval_time": eval_time,
                "eval_tps": eval_tps,
                "total_time": total_time,
                "pipeline_tps": n / total_time,
                "passed": passed_count,
                "completions": completions_with_status,
            }

            # Save to pending_combine queue for combiner
            temp_str = f"{temp:.1f}".replace('.', '_')
            result_file = f"{pending_combine_dir}/q{question_idx}_t{temp_str}_gpu{gpu_id}.json"
            temp_file = result_file + ".tmp"

            with open(temp_file, "w") as f:
                json.dump(result, f)
            os.rename(temp_file, result_file)  # Atomic write

            # Remove processing file
            os.remove(processing_file)

            processed_count += 1
            print(f"[EVAL {evaluator_id}] Q{question_idx} T={temp}: {passed_count}/{n} passed ({eval_tps:.0f} eval/s)", file=sys.stderr)

        except Exception as e:
            print(f"[EVAL {evaluator_id}] Error processing {processing_file}: {e}", file=sys.stderr)
            # Move back to pending for retry
            try:
                os.rename(processing_file, pending_file)
            except:
                pass

print(f"[EVAL {evaluator_id}] Shutting down, processed {processed_count} batches", file=sys.stderr)
