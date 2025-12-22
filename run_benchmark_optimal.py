#!/usr/bin/env python3
"""Optimal benchmark: Separate GPU generators and CPU evaluators for maximum parallelism.
   - GPU generators: Only generate, never wait for eval
   - CPU evaluators: Only evaluate, process pending queue
   Self-healing: automatically restarts crashed workers"""
import os
import sys
import subprocess
import argparse
import time
import json
import signal
import glob
import threading
from pathlib import Path
from collections import defaultdict

TEMPERATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7, 2.0]
NUM_GPUS = 8
NUM_EVALUATORS = 16  # Number of CPU evaluator processes
GPU_OFFSET = 0
STAGGER_DELAY = 0.5  # seconds between GPU launches
MAX_RESTARTS = 3  # max restarts per worker before giving up


def launch_gpu_generator(gpu_id, args, output_dir, temps_str, n_per_gpu):
    """Launch a GPU generator worker"""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id + GPU_OFFSET)

    cmd = [
        'python3', '/workspace/orkspace/gpu_generator.py',
        str(gpu_id),
        str(args.start),
        str(args.end),
        str(n_per_gpu),
        output_dir,
        temps_str
    ]

    log_file = open(f"{output_dir}/gpu{gpu_id}.log", 'a')
    p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return p, log_file


def launch_cpu_evaluator(eval_id, output_dir, num_workers=24):
    """Launch a CPU evaluator worker"""
    cmd = [
        'python3', '/workspace/orkspace/cpu_evaluator.py',
        str(eval_id),
        output_dir,
        str(num_workers)
    ]

    log_file = open(f"{output_dir}/eval{eval_id}.log", 'a')
    p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return p, log_file


def combiner_loop(output_dir, num_gpus, stop_event, combined_count):
    """Background thread: continuously combines GPU results as they complete.

    Input:  pending_combine/q{idx}_t{temp}_gpu{id}.json (8 files × 1250 each)
    Output: q{idx}/t{temp}_combined.json (1 file × 10000)
    """
    pending_combine_dir = f"{output_dir}/pending_combine"
    print(f"[COMBINER] Started - watching {pending_combine_dir}", flush=True)

    while not stop_event.is_set():
        # Find all evaluated files in pending_combine
        all_files = [f for f in glob.glob(f"{pending_combine_dir}/q*_t*_gpu*.json")
                     if not f.endswith('.tmp')]

        if not all_files:
            time.sleep(1)
            continue

        # Group by question+temperature: {(q_idx, temp_str): [files]}
        groups = defaultdict(list)
        for f in all_files:
            fname = os.path.basename(f)
            # Parse q{idx}_t{temp}_gpu{id}.json - e.g., q0_t0_1_gpu0.json
            # Format: q{idx}_t{major}_{minor}_gpu{id}.json
            import re
            match = re.match(r'q(\d+)_t(\d+_\d+)_gpu(\d+)\.json', fname)
            if not match:
                continue
            q_idx = int(match.group(1))
            temp_str = 't' + match.group(2)  # t0_1
            groups[(q_idx, temp_str)].append(f)

        for (q_idx, temp_str), files in groups.items():
            if stop_event.is_set():
                break

            # Skip if not all GPUs present
            if len(files) < num_gpus:
                continue

            # Create output directory
            q_dir = f"{output_dir}/q{q_idx}"
            os.makedirs(q_dir, exist_ok=True)
            combined_file = f"{q_dir}/{temp_str}_combined.json"

            # Skip if already combined
            if os.path.exists(combined_file):
                # Clean up source files
                for f in files:
                    try:
                        os.remove(f)
                    except:
                        pass
                continue

            # Combine all GPU files
            all_completions = []
            total_passed = 0
            total_tokens = 0
            total_gen_time = 0
            total_eval_time = 0
            first_data = None

            try:
                for f in sorted(files):
                    with open(f) as fp:
                        data = json.load(fp)
                    if first_data is None:
                        first_data = data
                    all_completions.extend(data.get('completions', []))
                    total_passed += data.get('passed', 0)
                    total_tokens += data.get('total_tokens', 0)
                    total_gen_time = max(total_gen_time, data.get('gen_time', 0))
                    total_eval_time = max(total_eval_time, data.get('eval_time', 0))
            except Exception as e:
                continue

            if first_data is None or not all_completions:
                continue

            n_total = len(all_completions)
            total_time = total_gen_time + total_eval_time

            combined = {
                "model": first_data.get("model"),
                "prompt": first_data.get("prompt"),
                "sampling_params": first_data.get("sampling_params"),
                "model_config": first_data.get("model_config"),
                "question_idx": q_idx,
                "question_title": first_data.get("question_title"),
                "temperature": first_data.get("temperature"),
                "n": n_total,
                "total_tokens": total_tokens,
                "gen_time": total_gen_time,
                "gen_tps": total_tokens / total_gen_time if total_gen_time > 0 else 0,
                "eval_time": total_eval_time,
                "eval_tps": n_total / total_eval_time if total_eval_time > 0 else 0,
                "total_time": total_time,
                "pipeline_tps": n_total / total_time if total_time > 0 else 0,
                "passed": total_passed,
                "pass_rate": 100 * total_passed / n_total if n_total > 0 else 0,
                "completions": all_completions,
            }

            # Write combined file
            temp_file = combined_file + ".tmp"
            with open(temp_file, 'w') as fp:
                json.dump(combined, fp)
            os.rename(temp_file, combined_file)

            # Remove source files from pending_combine
            for f in files:
                try:
                    os.remove(f)
                except:
                    pass

            combined_count[0] += 1
            print(f"[COMBINER] Q{q_idx} T={first_data.get('temperature')}: {total_passed}/{n_total} passed ({100*total_passed/n_total:.1f}%) - {combined_count[0]} combined", flush=True)

            # Check if all 16 temps are combined for this question -> create summary
            existing_combined = glob.glob(f"{q_dir}/t*_combined.json")
            summary_file = f"{q_dir}/summary.json"
            if len(existing_combined) >= len(TEMPERATURES) and not os.path.exists(summary_file):
                try:
                    summary_results = []
                    for cf in sorted(existing_combined):
                        with open(cf) as fp:
                            data = json.load(fp)
                        summary_results.append({
                            "temperature": data.get("temperature"),
                            "n": data.get("n"),
                            "passed": data.get("passed"),
                            "pass_rate": data.get("pass_rate"),
                            "gen_tps": data.get("gen_tps"),
                        })

                    # Sort by temperature
                    summary_results.sort(key=lambda x: x.get("temperature", 0))

                    summary = {
                        "question_idx": q_idx,
                        "question_title": summary_results[0].get("question_title", "") if summary_results else "",
                        "n": summary_results[0].get("n") if summary_results else 0,
                        "temperatures": [r["temperature"] for r in summary_results],
                        "results": summary_results,
                    }
                    with open(summary_file, 'w') as fp:
                        json.dump(summary, fp, indent=2)

                    # Find best temperature
                    best = max(summary_results, key=lambda x: x.get("pass_rate", 0))
                    print(f"[COMBINER] Q{q_idx} COMPLETE: best T={best['temperature']} ({best['pass_rate']:.1f}%)", flush=True)
                except Exception as e:
                    pass

        time.sleep(2)  # Check every 2 seconds

    print(f"[COMBINER] Stopped, combined {combined_count[0]} files total", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Optimal benchmark - separate GPU gen and CPU eval")
    parser.add_argument('--start', type=int, default=0, help='Start question index')
    parser.add_argument('--end', type=int, default=175, help='End question index')
    parser.add_argument('--n', type=int, default=10000, help='Total completions per temperature')
    parser.add_argument('--output_dir', '-o', type=str, default='results_optimal', help='Output directory')
    parser.add_argument('--num_evaluators', type=int, default=NUM_EVALUATORS, help='Number of CPU evaluator processes')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/pending_eval", exist_ok=True)
    os.makedirs(f"{output_dir}/pending_combine", exist_ok=True)

    total_questions = args.end - args.start
    n_per_gpu = args.n // NUM_GPUS
    temps_str = ','.join(str(t) for t in TEMPERATURES)

    print(f"{'='*70}")
    print(f"OPTIMAL BENCHMARK (separate GPU gen + CPU eval)")
    print(f"{'='*70}")
    print(f"Questions: {args.start} to {args.end} ({total_questions} total)")
    print(f"Temperatures: {len(TEMPERATURES)}")
    print(f"Completions: {args.n} total ({n_per_gpu} per GPU)")
    print(f"GPU Generators: {NUM_GPUS} (stagger: {STAGGER_DELAY}s)")
    print(f"CPU Evaluators: {args.num_evaluators}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    global_start = time.time()

    # Track workers
    gpu_workers = {}  # {gpu_id: (process, log_file, restart_count)}
    eval_workers = {}  # {eval_id: (process, log_file, restart_count)}
    gpu_completed = set()
    gpu_failed = set()
    eval_running = set()

    # Launch GPU generators with stagger delay
    print("\nLaunching GPU generators...")
    for gpu_id in range(NUM_GPUS):
        p, log_file = launch_gpu_generator(gpu_id, args, output_dir, temps_str, n_per_gpu)
        gpu_workers[gpu_id] = (p, log_file, 0)
        print(f"  [GPU {gpu_id}] Started generator")
        time.sleep(STAGGER_DELAY)

    # Launch CPU evaluators
    print(f"\nLaunching {args.num_evaluators} CPU evaluators...")
    for eval_id in range(args.num_evaluators):
        # Each evaluator gets 24 // num_evaluators workers, minimum 6
        workers_per_eval = max(6, 24 // args.num_evaluators)
        p, log_file = launch_cpu_evaluator(eval_id, output_dir, workers_per_eval)
        eval_workers[eval_id] = (p, log_file, 0)
        eval_running.add(eval_id)
        print(f"  [EVAL {eval_id}] Started evaluator (workers={workers_per_eval})")

    # Launch combiner thread (runs in parallel, combines results as they complete)
    print(f"\nLaunching combiner thread...")
    combiner_stop = threading.Event()
    combined_count = [0]  # Use list for mutable reference
    combiner_thread = threading.Thread(
        target=combiner_loop,
        args=(output_dir, NUM_GPUS, combiner_stop, combined_count),
        daemon=True
    )
    combiner_thread.start()

    print(f"\n{'='*70}")
    print(f"All workers launched!")
    print(f"GPU generators will run until all questions processed")
    print(f"CPU evaluators will run until all pending items processed")
    print(f"{'='*70}")
    print(f"\nMonitor:")
    print(f"  GPU progress: tail -f {output_dir}/gpu*.log")
    print(f"  Eval progress: tail -f {output_dir}/eval*.log")
    print(f"  Pending eval:    ls {output_dir}/pending_eval/ | wc -l")
    print(f"  Pending combine: ls {output_dir}/pending_combine/ | wc -l")
    print(f"  GPU status: nvidia-smi")
    print()

    # Monitor loop
    while len(gpu_completed) + len(gpu_failed) < NUM_GPUS or eval_running:
        time.sleep(5)

        # Check GPU generators
        for gpu_id in list(gpu_workers.keys()):
            if gpu_id in gpu_completed or gpu_id in gpu_failed:
                continue

            p, log_file, restart_count = gpu_workers[gpu_id]
            ret = p.poll()

            if ret is None:
                continue

            if ret == 0:
                log_file.close()
                gpu_completed.add(gpu_id)
                print(f"[GPU {gpu_id}] Completed successfully")
            else:
                log_file.close()
                if restart_count < MAX_RESTARTS:
                    print(f"[GPU {gpu_id}] Crashed (code {ret}), restarting ({restart_count + 1}/{MAX_RESTARTS})...")
                    time.sleep(STAGGER_DELAY)
                    p, log_file = launch_gpu_generator(gpu_id, args, output_dir, temps_str, n_per_gpu)
                    gpu_workers[gpu_id] = (p, log_file, restart_count + 1)
                else:
                    print(f"[GPU {gpu_id}] Failed after {MAX_RESTARTS} restarts")
                    gpu_failed.add(gpu_id)

        # Check if all GPUs done and no pending files - stop evaluators
        if len(gpu_completed) + len(gpu_failed) >= NUM_GPUS:
            pending_eval_count = len(list(Path(f"{output_dir}/pending_eval").glob("*.json")))
            if pending_eval_count == 0 and eval_running:
                print("\nAll generation complete and queue empty, stopping evaluators...")
                for eval_id in list(eval_running):
                    p, log_file, _ = eval_workers[eval_id]
                    p.terminate()
                    p.wait()
                    log_file.close()
                    print(f"[EVAL {eval_id}] Stopped")
                eval_running.clear()

        # Check evaluator health (restart if crashed while GPUs still generating)
        if len(gpu_completed) + len(gpu_failed) < NUM_GPUS:
            for eval_id in list(eval_running):
                p, log_file, restart_count = eval_workers[eval_id]
                ret = p.poll()
                if ret is not None and ret != 0:
                    log_file.close()
                    if restart_count < MAX_RESTARTS:
                        print(f"[EVAL {eval_id}] Crashed, restarting...")
                        workers_per_eval = max(6, 24 // args.num_evaluators)
                        p, log_file = launch_cpu_evaluator(eval_id, output_dir, workers_per_eval)
                        eval_workers[eval_id] = (p, log_file, restart_count + 1)
                    else:
                        print(f"[EVAL {eval_id}] Failed after {MAX_RESTARTS} restarts")
                        eval_running.discard(eval_id)

    global_time = time.time() - global_start

    print(f"\n{'='*70}")
    print(f"GENERATION & EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {global_time:.0f}s ({global_time/3600:.2f}h)")
    print(f"Questions: {total_questions}")
    print(f"Temps per question: {len(TEMPERATURES)}")
    print(f"GPUs completed: {len(gpu_completed)}/{NUM_GPUS}")
    if gpu_failed:
        print(f"GPUs failed: {sorted(gpu_failed)}")

    # Give combiner a few seconds to finish any remaining work
    print(f"\nWaiting for combiner to finish...")
    time.sleep(5)
    combiner_stop.set()
    combiner_thread.join(timeout=10)

    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Combined files: {combined_count[0]}")

    # Save summary
    with open(f"{output_dir}/run_summary.json", 'w') as f:
        json.dump({
            'start_idx': args.start,
            'end_idx': args.end,
            'n': args.n,
            'n_per_gpu': n_per_gpu,
            'temperatures': TEMPERATURES,
            'num_gpus': NUM_GPUS,
            'num_evaluators': args.num_evaluators,
            'stagger_delay': STAGGER_DELAY,
            'gpus_completed': sorted(gpu_completed),
            'gpus_failed': sorted(gpu_failed),
            'total_time_seconds': global_time,
        }, f, indent=2)


if __name__ == "__main__":
    main()
