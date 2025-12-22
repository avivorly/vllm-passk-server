#!/usr/bin/env python3
"""
Try to clear NVIDIA persistence daemon's stale process records.
This is the most likely culprit for zombie GPU memory allocations.
"""

import os
import sys
import subprocess
import time

print("=" * 80)
print("NVIDIA Persistence Daemon Cleanup")
print("=" * 80)

# The issue is likely that nvidia-persistenced is holding onto the process records
# even though the processes are dead. Let's try various approaches.

print("\n[Info] Checking NVIDIA persistence daemon status...")
try:
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'nvidia' in line.lower() and 'persist' in line.lower():
            print(f"  {line}")
except Exception as e:
    print(f"  Error: {e}")

print("\n[Attempt 1] Using pynvml to disable then re-enable persistence mode...")
print("  (This might force the daemon to clear stale records)")
try:
    import pynvml

    pynvml.nvmlInit()

    for gpu_id in [0, 1]:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        print(f"\nGPU {gpu_id}:")

        # Check current persistence mode
        try:
            current_mode = pynvml.nvmlDeviceGetPersistenceMode(handle)
            print(f"  Current persistence mode: {current_mode}")

            # Try to disable persistence mode
            print(f"  Attempting to disable persistence mode...")
            try:
                pynvml.nvmlDeviceSetPersistenceMode(handle, 0)
                print(f"    Disabled!")
                time.sleep(1)

                # Re-enable it
                print(f"  Re-enabling persistence mode...")
                pynvml.nvmlDeviceSetPersistenceMode(handle, 1)
                print(f"    Re-enabled!")

            except pynvml.NVMLError_NoPermission:
                print(f"    No permission (requires root)")
            except Exception as e:
                print(f"    Error: {e}")

        except Exception as e:
            print(f"  Error checking persistence mode: {e}")

    pynvml.nvmlShutdown()

except Exception as e:
    print(f"Failed: {e}")

print("\n[Attempt 2] Checking for nvidia-cuda-mps-control...")
mps_control_paths = [
    '/usr/bin/nvidia-cuda-mps-control',
    '/usr/local/cuda/bin/nvidia-cuda-mps-control',
    'nvidia-cuda-mps-control'
]

mps_found = False
for path in mps_control_paths:
    try:
        result = subprocess.run(['which', path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Found: {result.stdout.strip()}")
            mps_found = True
            break
    except:
        pass

if not mps_found:
    # Try running it directly
    try:
        result = subprocess.run(['nvidia-cuda-mps-control', '-d'],
                              capture_output=True, text=True, timeout=5)
        print(f"  nvidia-cuda-mps-control output: {result.stdout}")
        print(f"  nvidia-cuda-mps-control error: {result.stderr}")
    except FileNotFoundError:
        print("  nvidia-cuda-mps-control not found")
    except Exception as e:
        print(f"  Error running nvidia-cuda-mps-control: {e}")

print("\n[Attempt 3] Creating a process that attaches to GPU then exits...")
print("  (This might trigger cleanup of stale process records)")

try:
    import torch
    if torch.cuda.is_available():
        for gpu_id in [0, 1]:
            print(f"\nGPU {gpu_id}:")

            # Create a subprocess that will use the GPU
            script = f"""
import torch
torch.cuda.set_device({gpu_id})
x = torch.zeros(100, device='cuda:{gpu_id}')
print(f"Process attached to GPU {gpu_id}")
del x
torch.cuda.empty_cache()
"""
            # Write temporary script
            with open('/tmp/gpu_attach.py', 'w') as f:
                f.write(script)

            # Run it
            result = subprocess.run([sys.executable, '/tmp/gpu_attach.py'],
                                  capture_output=True, text=True, timeout=10)
            print(f"  Output: {result.stdout.strip()}")
            if result.stderr:
                print(f"  Error: {result.stderr.strip()}")

except Exception as e:
    print(f"Failed: {e}")

print("\n[Attempt 4] Using ctypes to call CUDA driver API directly...")
try:
    import ctypes

    # Try to load CUDA driver library
    cuda_paths = [
        'libcuda.so',
        'libcuda.so.1',
        '/usr/lib/x86_64-linux-gnu/libcuda.so',
        '/usr/lib/x86_64-linux-gnu/libcuda.so.1',
    ]

    cuda = None
    for path in cuda_paths:
        try:
            cuda = ctypes.CDLL(path)
            print(f"  Loaded CUDA driver from: {path}")
            break
        except:
            continue

    if cuda:
        # Initialize CUDA
        print("  Initializing CUDA driver...")
        result = cuda.cuInit(0)
        print(f"  cuInit result: {result}")

        if result == 0:  # CUDA_SUCCESS
            # Get device handle
            for gpu_id in [0, 1]:
                device = ctypes.c_int()
                result = cuda.cuDeviceGet(ctypes.byref(device), gpu_id)
                print(f"\n  GPU {gpu_id}: cuDeviceGet result: {result}")

                if result == 0:
                    # Create context
                    context = ctypes.c_void_p()
                    result = cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)
                    print(f"    cuCtxCreate result: {result}")

                    if result == 0:
                        # Synchronize
                        result = cuda.cuCtxSynchronize()
                        print(f"    cuCtxSynchronize result: {result}")

                        # Destroy context
                        result = cuda.cuCtxDestroy_v2(context)
                        print(f"    cuCtxDestroy result: {result}")
    else:
        print("  Could not load CUDA driver library")

except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[Attempt 5] Check if processes have parent PIDs we can kill...")
try:
    import pynvml
    pynvml.nvmlInit()

    zombie_pids = [3583562, 3586474]

    # Check all running processes for parents
    result = subprocess.run(['ps', 'axo', 'pid,ppid,cmd'],
                          capture_output=True, text=True)

    print("  Searching for parent processes...")
    for line in result.stdout.split('\n'):
        for zpid in zombie_pids:
            if str(zpid) in line:
                print(f"    Found reference: {line}")

    pynvml.nvmlShutdown()

except Exception as e:
    print(f"Failed: {e}")

print("\n[Attempt 6] Try nvidia-smi compute-apps flag...")
try:
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                           '--format=csv,noheader'],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Compute apps:\n{result.stdout}")
    else:
        print(f"  Error: {result.stderr}")
except Exception as e:
    print(f"Failed: {e}")

print("\n" + "=" * 80)
print("NVIDIA daemon cleanup attempts complete")
print("=" * 80)
