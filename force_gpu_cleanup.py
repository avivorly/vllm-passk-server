#!/usr/bin/env python3
"""
Aggressive GPU memory cleanup focusing on zombie process cleanup.
"""

import sys
import os
import signal

print("=" * 80)
print("Aggressive GPU Memory Cleanup for Zombie Processes")
print("=" * 80)

# First, let's try to send signals to the zombie PIDs
zombie_pids = [3583562, 3586474]

print("\n[Attempt 1] Trying to send SIGKILL to zombie PIDs...")
for pid in zombie_pids:
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"  Sent SIGKILL to PID {pid}")
    except ProcessLookupError:
        print(f"  PID {pid} does not exist (expected for zombie)")
    except PermissionError:
        print(f"  No permission to kill PID {pid}")
    except Exception as e:
        print(f"  Failed to kill PID {pid}: {e}")

print("\n[Attempt 2] Using pynvml to get detailed GPU info...")
try:
    import pynvml

    pynvml.nvmlInit()

    for gpu_id in [0, 1]:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        print(f"\nGPU {gpu_id}:")

        # Get all process types
        try:
            compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)

            print(f"  Compute processes: {len(compute_procs)}")
            for p in compute_procs:
                print(f"    PID {p.pid}: {p.usedGpuMemory / 1024**2:.2f} MB")

            print(f"  Graphics processes: {len(graphics_procs)}")
            for p in graphics_procs:
                print(f"    PID {p.pid}: {p.usedGpuMemory / 1024**2:.2f} MB")

        except Exception as e:
            print(f"  Error getting processes: {e}")

        # Try to get accounting stats
        try:
            mode = pynvml.nvmlDeviceGetAccountingMode(handle)
            print(f"  Accounting mode: {mode}")
        except Exception as e:
            print(f"  Could not get accounting mode: {e}")

        # Check persistence mode
        try:
            persist = pynvml.nvmlDeviceGetPersistenceMode(handle)
            print(f"  Persistence mode: {persist}")
        except Exception as e:
            print(f"  Could not get persistence mode: {e}")

    pynvml.nvmlShutdown()

except Exception as e:
    print(f"Failed: {e}")

print("\n[Attempt 3] Trying to access /proc entries for zombie PIDs...")
for pid in zombie_pids:
    proc_path = f"/proc/{pid}"
    if os.path.exists(proc_path):
        print(f"\n  PID {pid} proc directory exists!")
        try:
            # Try to read status
            with open(f"{proc_path}/status", 'r') as f:
                status = f.read()
                print(f"    Status:\n{status[:500]}")
        except Exception as e:
            print(f"    Could not read status: {e}")

        try:
            # Try to read cmdline
            with open(f"{proc_path}/cmdline", 'r') as f:
                cmdline = f.read()
                print(f"    Cmdline: {cmdline}")
        except Exception as e:
            print(f"    Could not read cmdline: {e}")
    else:
        print(f"  PID {pid} proc directory does not exist")

print("\n[Attempt 4] Creating aggressive CUDA context reset script...")

# Try multiple CUDA context resets
try:
    import torch
    if torch.cuda.is_available():
        for gpu_id in [0, 1]:
            print(f"\nGPU {gpu_id}:")

            # Try to allocate and free maximum memory to flush any cached allocations
            try:
                torch.cuda.set_device(gpu_id)

                # Get available memory
                mem_free = torch.cuda.mem_get_info(gpu_id)[0]
                print(f"  Free memory: {mem_free / 1024**3:.2f} GB")

                # Try to allocate a large chunk to force cleanup
                try:
                    # Allocate 90% of free memory
                    size = int(mem_free * 0.9 / 4)  # float32 = 4 bytes
                    print(f"  Attempting to allocate {size * 4 / 1024**3:.2f} GB...")
                    x = torch.zeros(size, device=f'cuda:{gpu_id}', dtype=torch.float32)
                    del x
                    print(f"  Allocation and deletion successful")
                except RuntimeError as e:
                    print(f"  Could not allocate: {e}")

                # Multiple cache clears
                for i in range(5):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(gpu_id)

                # IPC collect
                torch.cuda.ipc_collect()

                # Check memory again
                mem_free_after = torch.cuda.mem_get_info(gpu_id)[0]
                print(f"  Free memory after: {mem_free_after / 1024**3:.2f} GB")
                print(f"  Change: {(mem_free_after - mem_free) / 1024**3:.2f} GB")

            except Exception as e:
                print(f"  Error: {e}")

except Exception as e:
    print(f"Failed: {e}")

print("\n[Attempt 5] Checking for CUDA IPC handles...")
try:
    import torch
    if torch.cuda.is_available():
        # Try to collect all CUDA IPC memory
        torch.cuda.ipc_collect()
        print("  torch.cuda.ipc_collect() called")

        # Check if there are any shared memory segments
        import subprocess
        result = subprocess.run(['ipcs', '-m'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n  Shared memory segments:")
            lines = result.stdout.split('\n')
            for line in lines[:20]:  # First 20 lines
                print(f"    {line}")
        else:
            print(f"  Could not list shared memory: {result.stderr}")

except Exception as e:
    print(f"Failed: {e}")

print("\n[Attempt 6] Checking CUDA MPS control...")
# Check if CUDA MPS is running
mps_pipe = os.environ.get('CUDA_MPS_PIPE_DIRECTORY', '/tmp/nvidia-mps')
print(f"  CUDA_MPS_PIPE_DIRECTORY: {mps_pipe}")

if os.path.exists(mps_pipe):
    print(f"  MPS pipe directory exists")
    try:
        import subprocess
        # Try to quit MPS (this might help)
        result = subprocess.run(['echo', 'quit'], capture_output=True)
        print(f"  Attempted MPS quit")
    except Exception as e:
        print(f"  Could not interact with MPS: {e}")
else:
    print(f"  MPS pipe directory does not exist (MPS not running)")

print("\n" + "=" * 80)
print("Aggressive cleanup complete. Checking GPU status...")
print("=" * 80)
