#!/usr/bin/env python3
"""
Final diagnostic to understand the zombie memory issue.
"""

import pynvml
import os
import time

print("=" * 80)
print("Final Diagnostic: Understanding Zombie Memory")
print("=" * 80)

pynvml.nvmlInit()

# Get all GPUs with memory usage
print("\n[Step 1] Scanning all GPUs for zombie processes...")
zombie_info = []

for gpu_id in range(8):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_gb = mem_info.used / 1024**3

    if used_gb > 10:  # Significant memory usage
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for proc in procs:
                pid = proc.pid
                mem_mb = proc.usedGpuMemory / 1024**2

                # Check if process exists
                exists = os.path.exists(f"/proc/{pid}")

                zombie_info.append({
                    'gpu_id': gpu_id,
                    'pid': pid,
                    'mem_mb': mem_mb,
                    'exists': exists
                })

                print(f"  GPU {gpu_id}: PID {pid}, {mem_mb:.0f} MB, " +
                      f"{'EXISTS' if exists else 'ZOMBIE'}")

        except Exception as e:
            print(f"  GPU {gpu_id}: Error - {e}")

print(f"\n[Step 2] Summary:")
print(f"  Total zombie processes found: {len([z for z in zombie_info if not z['exists']])}")
print(f"  Total active processes: {len([z for z in zombie_info if z['exists']])}")
print(f"  Total memory held by zombies: {sum(z['mem_mb'] for z in zombie_info if not z['exists']):.0f} MB")

# Try to understand WHY the memory is stuck
print("\n[Step 3] Understanding persistence mode...")
for gpu_id in range(8):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    persist = pynvml.nvmlDeviceGetPersistenceMode(handle)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"  GPU {gpu_id}: Persistence={'ON' if persist else 'OFF'}, " +
          f"Used={mem_info.used / 1024**3:.2f} GB")

print("\n[Step 4] The Problem:")
print("  When persistence mode is ENABLED (which requires root to change),")
print("  the nvidia-persistenced daemon keeps GPU contexts alive even after")
print("  processes die. This is the zombie memory we're seeing.")
print()
print("  Without root access, we CANNOT:")
print("  - Disable persistence mode")
print("  - Clear accounted PIDs")
print("  - Force GPU reset")
print("  - Restart nvidia-persistenced")

print("\n[Step 5] Potential workarounds (all attempted):")
checks = [
    ("cudaDeviceReset()", "Called successfully but didn't free memory"),
    ("torch.cuda.empty_cache()", "Only clears current process cache"),
    ("pynvml reset functions", "Require root permissions"),
    ("nvidia-smi --gpu-reset", "Requires root permissions"),
    ("nvidia-smi --clear-accounted-apps", "Requires root permissions"),
    ("Allocate/free large memory", "Doesn't evict zombie allocations"),
    ("CUDA context manipulation", "Can't affect dead process contexts"),
    ("Kill zombie PIDs", "Processes already dead"),
]

for method, result in checks:
    print(f"  ✗ {method}: {result}")

print("\n[Step 6] What WOULD work with root:")
print("  sudo nvidia-smi --clear-accounted-apps")
print("  OR")
print("  sudo nvidia-smi -i 0,1 --gpu-reset")
print("  OR")
print("  sudo nvidia-smi -pm 0  # Disable persistence")
print("  sudo nvidia-smi -pm 1  # Re-enable persistence")

print("\n[Step 7] Checking if we're in a container with limited privileges...")
in_container = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
print(f"  Running in container: {in_container}")

if in_container:
    print("  Container may have --privileged flag or --cap-add=SYS_ADMIN needed")

# Final memory snapshot
print("\n[Step 8] Final memory snapshot:")
for gpu_id in range(8):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

    used_gb = mem_info.used / 1024**3
    zombie_count = sum(1 for p in procs if not os.path.exists(f"/proc/{p.pid}"))
    active_count = sum(1 for p in procs if os.path.exists(f"/proc/{p.pid}"))

    if used_gb > 1:
        print(f"  GPU {gpu_id}: {used_gb:5.2f} GB used, " +
              f"{active_count} active procs, {zombie_count} zombies")

pynvml.nvmlShutdown()

print("\n" + "=" * 80)
print("CONCLUSION: Without root access, zombie GPU memory cannot be freed")
print("when persistence mode is enabled. The NVIDIA driver/daemon is holding")
print("onto dead process contexts.")
print("=" * 80)
