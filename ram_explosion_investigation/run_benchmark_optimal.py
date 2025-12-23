#!/usr/bin/env python3
"""Optimal benchmark: Separate GPU generators and CPU evaluators for maximum parallelism.
   - GPU generators: Only generate, never wait for eval
   - CPU evaluators: Only evaluate, process pending queue
   Self-healing: automatically restarts crashed workers
   RAM-aware: scales evaluators up/down based on memory pressure"""
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

# RAM thresholds for auto-scaling
RAM_HIGH_THRESHOLD = 95  # Kill evaluators when RAM > 95%
RAM_LOW_THRESHOLD = 80   # Restart evaluators when RAM < 80%
RAM_CHECK_INTERVAL = 5   # Check RAM every 5 seconds


def get_ram_usage_percent():
    """Get current RAM usage as percentage (0-100)"""
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()

        mem_info = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(':')
                value = int(parts[1])  # in kB
                mem_info[key] = value

        total = mem_info.get('MemTotal', 1)
        available = mem_info.get('MemAvailable', mem_info.get('MemFree', 0))
        used = total - available
        return 100.0 * used / total
    except Exception as e:
        return 50.0  # Default to 50% if can't read


def get_system_stats():
    """Get comprehensive system stats for monitoring"""
    import datetime
    stats = {"timestamp": datetime.datetime.now().isoformat()}

    # RAM stats
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        mem_info = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                mem_info[parts[0].rstrip(':')] = int(parts[1])

        total_gb = mem_info.get('MemTotal', 0) / 1024 / 1024
        avail_gb = mem_info.get('MemAvailable', 0) / 1024 / 1024
        used_gb = total_gb - avail_gb
        stats["ram_total_gb"] = round(total_gb, 1)
        stats["ram_used_gb"] = round(used_gb, 1)
        stats["ram_percent"] = round(100 * used_gb / total_gb, 1) if total_gb > 0 else 0
    except:
        stats["ram_error"] = "failed to read"

    # CPU stats
    try:
        with open('/proc/loadavg', 'r') as f:
            load = f.read().split()
        stats["cpu_load_1m"] = float(load[0])
        stats["cpu_load_5m"] = float(load[1])
        stats["cpu_load_15m"] = float(load[2])
    except:
        pass

    # GPU stats via nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                gpus.append({
                    "id": int(parts[0]),
                    "vram_used_mb": int(parts[1]),
                    "vram_total_mb": int(parts[2]),
                    "gpu_util_percent": int(parts[3]),
                    "temp_c": int(parts[4])
                })
        stats["gpus"] = gpus
        stats["vram_avg_percent"] = round(sum(g["vram_used_mb"]/g["vram_total_mb"]*100 for g in gpus)/len(gpus), 1) if gpus else 0
        stats["gpu_util_avg"] = round(sum(g["gpu_util_percent"] for g in gpus)/len(gpus), 1) if gpus else 0
    except:
        stats["gpu_error"] = "failed to read"

    # Process counts
    try:
        result = subprocess.run(['pgrep', '-f', 'gpu_generator'], capture_output=True, text=True)
        stats["gpu_generator_procs"] = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        result = subprocess.run(['pgrep', '-f', 'cpu_evaluator'], capture_output=True, text=True)
        stats["cpu_evaluator_procs"] = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    except:
        pass

    return stats


def get_top_memory_processes(limit=10):
    """Get top processes by memory usage"""
    try:
        result = subprocess.run(
            ['ps', 'aux', '--sort=-rss'],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')[1:limit+1]  # Skip header
        procs = []
        for line in lines:
            parts = line.split(None, 10)
            if len(parts) >= 11:
                procs.append({
                    "pid": parts[1],
                    "rss_mb": int(parts[5]) // 1024 if parts[5].isdigit() else 0,
                    "vsz_mb": int(parts[4]) // 1024 if parts[4].isdigit() else 0,
                    "cpu": parts[2],
                    "mem": parts[3],
                    "cmd": parts[10][:80]
                })
        return procs
    except:
        return []


def monitor_loop(output_dir, stop_event, interval=60):
    """Background thread: logs system stats.
    - Every 'interval' seconds normally
    - Every 1 second when RAM > 50%"""
    import datetime
    log_file = f"{output_dir}/system_monitor.log"

    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Monitor started at {datetime.datetime.now().isoformat()}\n")
        f.write(f"{'='*80}\n\n")

    last_log_time = 0
    high_ram_mode = False

    while not stop_event.is_set():
        try:
            stats = get_system_stats()
            ram_percent = stats.get('ram_percent', 0)

            # Determine logging interval: 1s if RAM > 50%, otherwise normal interval
            current_interval = 1 if ram_percent > 50 else interval
            now = time.time()

            # Check if we should log
            if now - last_log_time >= current_interval:
                last_log_time = now

                # Track mode changes
                if ram_percent > 50 and not high_ram_mode:
                    high_ram_mode = True
                    with open(log_file, 'a') as f:
                        f.write(f"\n>>> HIGH RAM MODE ACTIVATED (RAM > 50%) - logging every 1s <<<\n\n")
                elif ram_percent <= 50 and high_ram_mode:
                    high_ram_mode = False
                    with open(log_file, 'a') as f:
                        f.write(f"\n>>> Normal mode resumed (RAM <= 50%) - logging every {interval}s <<<\n\n")

                # Also get pending queue sizes
                try:
                    pending_eval = len(list(Path(f"{output_dir}/pending_eval").glob("*.json")))
                    pending_combine = len(list(Path(f"{output_dir}/pending_combine").glob("*.json")))
                    stats["pending_eval"] = pending_eval
                    stats["pending_combine"] = pending_combine
                except:
                    pass

                # Format log line with more data
                log_line = (
                    f"{stats.get('timestamp', 'N/A')} | "
                    f"RAM: {stats.get('ram_used_gb', '?')}/{stats.get('ram_total_gb', '?')}GB ({stats.get('ram_percent', '?')}%) | "
                    f"CPU: {stats.get('cpu_load_1m', '?')}/{stats.get('cpu_load_5m', '?')}/{stats.get('cpu_load_15m', '?')} | "
                    f"GPU util: {stats.get('gpu_util_avg', '?')}% | "
                    f"VRAM: {stats.get('vram_avg_percent', '?')}% | "
                    f"pending_eval: {stats.get('pending_eval', '?')} | "
                    f"pending_combine: {stats.get('pending_combine', '?')} | "
                    f"gpu_procs: {stats.get('gpu_generator_procs', '?')} | "
                    f"eval_procs: {stats.get('cpu_evaluator_procs', '?')}"
                )

                with open(log_file, 'a') as f:
                    f.write(log_line + "\n")

                    # In high RAM mode, also log top memory processes
                    if high_ram_mode:
                        top_procs = get_top_memory_processes(10)
                        if top_procs:
                            f.write("  TOP MEMORY PROCESSES:\n")
                            for p in top_procs:
                                f.write(f"    PID {p['pid']}: {p['rss_mb']}MB RSS, {p['mem']}% mem - {p['cmd']}\n")

                # Print to stdout occasionally (not every second in high RAM mode)
                if not high_ram_mode or (int(now) % 10 == 0):
                    print(f"[MONITOR] {log_line}", flush=True)

        except Exception as e:
            with open(log_file, 'a') as f:
                f.write(f"Monitor error: {e}\n")

        # Sleep 1 second between checks
        time.sleep(1)

    with open(log_file, 'a') as f:
        f.write(f"\nMonitor stopped at {datetime.datetime.now().isoformat()}\n")


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
                    question_title = ""
                    for cf in sorted(existing_combined):
                        with open(cf) as fp:
                            data = json.load(fp)
                        # Get question_title from first file
                        if not question_title:
                            question_title = data.get("question_title", "")
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
                        "question_title": question_title,
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
    workers_per_eval = max(6, 24 // args.num_evaluators)  # Original setting
    for eval_id in range(args.num_evaluators):
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

    # Launch system monitor thread (logs RAM, VRAM, CPU every 60s)
    print(f"Launching system monitor (logs to {output_dir}/system_monitor.log)...")
    monitor_stop = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_loop,
        args=(output_dir, monitor_stop, 60),
        daemon=True
    )
    monitor_thread.start()

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

    # Track suspended evaluators for RAM-based scaling
    eval_suspended = set()  # eval_ids that are suspended due to high RAM
    last_ram_check = time.time()

    # Monitor loop
    while len(gpu_completed) + len(gpu_failed) < NUM_GPUS or eval_running:
        time.sleep(5)

        # RAM-based auto-scaling: check every RAM_CHECK_INTERVAL seconds
        if time.time() - last_ram_check >= RAM_CHECK_INTERVAL:
            last_ram_check = time.time()
            ram_pct = get_ram_usage_percent()

            if ram_pct > RAM_HIGH_THRESHOLD and eval_running:
                # RAM critical! Kill some evaluators
                running_list = sorted(eval_running - eval_suspended)
                if running_list:
                    # Kill half of running evaluators
                    to_kill = running_list[:max(1, len(running_list) // 2)]
                    for eval_id in to_kill:
                        if eval_id in eval_workers:
                            p, log_file, _ = eval_workers[eval_id]
                            try:
                                p.terminate()
                                p.wait(timeout=5)
                            except:
                                try:
                                    p.kill()
                                except:
                                    pass
                            log_file.close()
                            eval_suspended.add(eval_id)
                            eval_running.discard(eval_id)
                    print(f"[RAM CRITICAL: {ram_pct:.1f}%] Suspended {len(to_kill)} evaluators: {to_kill}", flush=True)

            elif ram_pct < RAM_LOW_THRESHOLD and eval_suspended:
                # RAM is low, restart some suspended evaluators
                to_restart = list(eval_suspended)[:max(1, len(eval_suspended) // 2)]
                for eval_id in to_restart:
                    p, log_file = launch_cpu_evaluator(eval_id, output_dir, workers_per_eval)
                    eval_workers[eval_id] = (p, log_file, 0)
                    eval_running.add(eval_id)
                    eval_suspended.discard(eval_id)
                print(f"[RAM OK: {ram_pct:.1f}%] Restarted {len(to_restart)} evaluators: {to_restart}", flush=True)

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
    monitor_stop.set()
    combiner_thread.join(timeout=10)
    monitor_thread.join(timeout=5)

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
