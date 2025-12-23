#!/usr/bin/env python3
"""CPU Evaluator: Watches pending queue, evaluates completions, writes final results.
   Multiple evaluators can run in parallel for high CPU utilization.
   Memory-aware: explicitly cleans up after each file to prevent RAM explosion.
   ENHANCED LOGGING: Captures detailed metrics for debugging RAM explosions."""
import os
import sys
import json
import time
import glob
import signal
import datetime
import gc
import subprocess

# Get args: evaluator_id output_dir num_workers
evaluator_id = int(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])
num_eval_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 24

pending_eval_dir = f"{output_dir}/pending_eval"
pending_combine_dir = f"{output_dir}/pending_combine"
os.makedirs(pending_combine_dir, exist_ok=True)

# Setup detailed logging
eval_log_file = f"{output_dir}/eval{evaluator_id}_detailed.log"

def log_detail(msg):
    """Log detailed info for debugging"""
    timestamp = datetime.datetime.now().isoformat()
    with open(eval_log_file, 'a') as f:
        f.write(f"{timestamp} | {msg}\n")

def get_memory_mb():
    """Get current process RSS in MB"""
    try:
        with open(f'/proc/{os.getpid()}/statm', 'r') as f:
            pages = int(f.read().split()[1])  # RSS in pages
        return pages * 4096 / 1024 / 1024  # Convert to MB
    except:
        return -1

def get_child_process_count():
    """Count child processes of this evaluator"""
    try:
        result = subprocess.run(['pgrep', '-P', str(os.getpid())],
                                capture_output=True, text=True, timeout=2)
        return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    except:
        return -1

def get_system_ram_gb():
    """Get total system RAM usage in GB"""
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        mem_info = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                mem_info[parts[0].rstrip(':')] = int(parts[1])
        total = mem_info.get('MemTotal', 0) / 1024 / 1024
        avail = mem_info.get('MemAvailable', 0) / 1024 / 1024
        return round(total - avail, 1), round(total, 1)
    except:
        return -1, -1

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
mem_before_dataset = get_memory_mb()
dataset = load_code_generation_dataset(release_version="v6")
mem_after_dataset = get_memory_mb()
print(f"[EVAL {evaluator_id}] Dataset loaded ({len(dataset)} problems)", file=sys.stderr)

# Pre-compute test data sizes for all questions
test_data_sizes = {}
for q_idx in range(len(dataset)):
    try:
        sample = dataset[q_idx].get_evaluation_sample()
        io = json.loads(sample['input_output'])
        input_size = sum(len(str(x)) for x in io['inputs'])
        output_size = sum(len(str(x)) for x in io['outputs'])
        num_tests = len(io['inputs'])
        test_data_sizes[q_idx] = {'input_bytes': input_size, 'output_bytes': output_size,
                                   'total_bytes': input_size + output_size, 'num_tests': num_tests}
    except:
        test_data_sizes[q_idx] = {'error': True}

log_detail(f"STARTUP | mem_before_dataset={mem_before_dataset:.0f}MB | mem_after_dataset={mem_after_dataset:.0f}MB | workers={num_eval_workers}")

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
            # === BEFORE LOAD ===
            mem_before_load = get_memory_mb()
            children_before = get_child_process_count()
            sys_ram_before, sys_ram_total = get_system_ram_gb()

            with open(processing_file) as f:
                data = json.load(f)

            file_size_kb = os.path.getsize(processing_file) / 1024

            question_idx = data["question_idx"]
            temp = data["temperature"]
            gpu_id = data["gpu_id"]
            completions = data["completions"]
            n = data["n"]

            # Get test data size info
            test_info = test_data_sizes.get(question_idx, {})
            test_bytes = test_info.get('total_bytes', -1)
            num_tests = test_info.get('num_tests', -1)

            # Get evaluation sample
            problem = dataset[question_idx]
            eval_sample = problem.get_evaluation_sample()

            # === BEFORE EVAL ===
            mem_before_eval = get_memory_mb()

            log_detail(f"BEFORE_EVAL | Q{question_idx} T={temp} | "
                       f"file_kb={file_size_kb:.0f} | test_bytes={test_bytes} | num_tests={num_tests} | "
                       f"mem_self={mem_before_eval:.0f}MB | children={children_before} | "
                       f"sys_ram={sys_ram_before}/{sys_ram_total}GB")

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

            # === AFTER EVAL ===
            mem_after_eval = get_memory_mb()
            children_after = get_child_process_count()
            sys_ram_after, _ = get_system_ram_gb()

            passed_count = sum(1 for r in results[0] if all(x == True for x in r))
            pass_list = [all(x == True for x in r) for r in results[0]]

            log_detail(f"AFTER_EVAL | Q{question_idx} T={temp} | "
                       f"passed={passed_count}/{n} | eval_time={eval_time:.1f}s | tps={eval_tps:.0f} | "
                       f"mem_self={mem_after_eval:.0f}MB (Δ{mem_after_eval-mem_before_eval:+.0f}) | "
                       f"children={children_after} | sys_ram={sys_ram_after}GB (Δ{sys_ram_after-sys_ram_before:+.1f})")

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

            # CRITICAL: Explicit memory cleanup to prevent RAM explosion
            # These objects hold MB of data (1250 completions × large strings)
            del data, completions, results, metrics
            del completions_with_status, result, pass_list
            del eval_sample, problem
            gc.collect()  # Force garbage collection

            # === AFTER CLEANUP ===
            mem_after_cleanup = get_memory_mb()
            children_after_cleanup = get_child_process_count()
            sys_ram_after_cleanup, _ = get_system_ram_gb()

            log_detail(f"AFTER_CLEANUP | Q{question_idx} T={temp} | "
                       f"mem_self={mem_after_cleanup:.0f}MB (Δ{mem_after_cleanup-mem_before_load:+.0f} from start) | "
                       f"children={children_after_cleanup} | sys_ram={sys_ram_after_cleanup}GB | "
                       f"processed_count={processed_count}")

        except Exception as e:
            log_detail(f"ERROR | {processing_file} | {e}")
            print(f"[EVAL {evaluator_id}] Error processing {processing_file}: {e}", file=sys.stderr)
            # Move back to pending for retry
            try:
                os.rename(processing_file, pending_file)
            except:
                pass

final_mem = get_memory_mb()
final_sys_ram, _ = get_system_ram_gb()
log_detail(f"SHUTDOWN | processed_count={processed_count} | final_mem_self={final_mem:.0f}MB | final_sys_ram={final_sys_ram}GB")
print(f"[EVAL {evaluator_id}] Shutting down, processed {processed_count} batches", file=sys.stderr)
