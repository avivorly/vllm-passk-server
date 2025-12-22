#!/usr/bin/env python3
"""
Comprehensive GPU memory cleanup script using multiple approaches.
Tries various methods to free GPU memory without root access.
"""

import sys
import os
import gc
import time

print("=" * 80)
print("GPU Memory Cleanup Script")
print("=" * 80)

# Method 1: Try pynvml (nvidia-ml-py)
print("\n[Method 1] Attempting to use pynvml/nvidia-ml-py...")
try:
    import pynvml

    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"Found {device_count} GPUs")

    for i in [0, 1]:  # Focus on GPUs 0 and 1
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"\nGPU {i} ({name}):")
            print(f"  Total: {mem_info.total / 1024**3:.2f} GB")
            print(f"  Used: {mem_info.used / 1024**3:.2f} GB")
            print(f"  Free: {mem_info.free / 1024**3:.2f} GB")

            # Try to get compute processes
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                print(f"  Running processes: {len(procs)}")
                for proc in procs:
                    print(f"    PID: {proc.pid}, Memory: {proc.usedGpuMemory / 1024**2:.2f} MB")
            except Exception as e:
                print(f"  Could not get process info: {e}")

            # Try to reset GPU (this usually requires root, but worth trying)
            print(f"  Attempting nvmlDeviceResetGpuLockedClocks...")
            try:
                pynvml.nvmlDeviceResetGpuLockedClocks(handle)
                print(f"    Success!")
            except pynvml.NVMLError_NoPermission:
                print(f"    No permission (expected without root)")
            except Exception as e:
                print(f"    Failed: {e}")

            # Try to reset application clocks
            print(f"  Attempting nvmlDeviceResetApplicationsClocks...")
            try:
                pynvml.nvmlDeviceResetApplicationsClocks(handle)
                print(f"    Success!")
            except pynvml.NVMLError_NoPermission:
                print(f"    No permission (expected without root)")
            except Exception as e:
                print(f"    Failed: {e}")

        except Exception as e:
            print(f"Error with GPU {i}: {e}")

    pynvml.nvmlShutdown()
    print("\n[Method 1] pynvml check complete")

except ImportError:
    print("[Method 1] pynvml not installed. Installing...")
    os.system("pip install nvidia-ml-py3 -q")
    print("Please run this script again after installation.")
    sys.exit(1)
except Exception as e:
    print(f"[Method 1] Failed: {e}")

# Method 2: Try PyTorch CUDA operations
print("\n[Method 2] Attempting PyTorch CUDA operations...")
try:
    import torch

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} devices")

        for i in [0, 1]:
            print(f"\nGPU {i}:")
            try:
                # Set device
                torch.cuda.set_device(i)

                # Get memory stats before
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  Allocated: {mem_allocated:.2f} GB")
                print(f"  Reserved: {mem_reserved:.2f} GB")

                # Try various cleanup operations
                print(f"  Attempting torch.cuda.empty_cache()...")
                torch.cuda.empty_cache()

                print(f"  Attempting torch.cuda.synchronize()...")
                torch.cuda.synchronize(i)

                print(f"  Attempting torch.cuda.reset_peak_memory_stats()...")
                torch.cuda.reset_peak_memory_stats(i)

                print(f"  Attempting torch.cuda.reset_accumulated_memory_stats()...")
                torch.cuda.reset_accumulated_memory_stats(i)

                # Try to reset max memory
                print(f"  Attempting torch.cuda.reset_max_memory_allocated()...")
                torch.cuda.reset_max_memory_allocated(i)
                torch.cuda.reset_max_memory_cached(i)

                # Get memory stats after
                mem_allocated_after = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved_after = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  After cleanup:")
                print(f"    Allocated: {mem_allocated_after:.2f} GB")
                print(f"    Reserved: {mem_reserved_after:.2f} GB")

                # Try to force IPC cleanup
                print(f"  Attempting torch.cuda.ipc_collect()...")
                torch.cuda.ipc_collect()

            except Exception as e:
                print(f"  Error: {e}")

        print("\n[Method 2] PyTorch cleanup complete")
    else:
        print("[Method 2] CUDA not available in PyTorch")

except ImportError:
    print("[Method 2] PyTorch not installed")
except Exception as e:
    print(f"[Method 2] Failed: {e}")

# Method 3: Python garbage collection
print("\n[Method 3] Running Python garbage collection...")
try:
    collected = gc.collect()
    print(f"  Collected {collected} objects")
    print("[Method 3] Garbage collection complete")
except Exception as e:
    print(f"[Method 3] Failed: {e}")

# Method 4: Check for any CUDA context we can manipulate
print("\n[Method 4] Attempting CUDA context manipulation...")
try:
    import torch
    if torch.cuda.is_available():
        for i in [0, 1]:
            try:
                with torch.cuda.device(i):
                    # Create a small tensor and immediately delete it
                    # This might help establish and clear a fresh context
                    x = torch.zeros(1, device=f'cuda:{i}')
                    del x
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(i)
                print(f"  GPU {i}: Context manipulation attempted")
            except Exception as e:
                print(f"  GPU {i}: Failed - {e}")
    print("[Method 4] Context manipulation complete")
except Exception as e:
    print(f"[Method 4] Failed: {e}")

# Method 5: Try CuPy if available
print("\n[Method 5] Attempting CuPy cleanup...")
try:
    import cupy as cp

    for i in [0, 1]:
        try:
            with cp.cuda.Device(i):
                # Get memory pool
                mempool = cp.get_default_memory_pool()
                print(f"  GPU {i}:")
                print(f"    Used bytes: {mempool.used_bytes() / 1024**3:.2f} GB")
                print(f"    Total bytes: {mempool.total_bytes() / 1024**3:.2f} GB")

                # Try to free all blocks
                print(f"    Attempting mempool.free_all_blocks()...")
                mempool.free_all_blocks()

                print(f"    After cleanup:")
                print(f"      Used bytes: {mempool.used_bytes() / 1024**3:.2f} GB")
                print(f"      Total bytes: {mempool.total_bytes() / 1024**3:.2f} GB")

        except Exception as e:
            print(f"  GPU {i}: Failed - {e}")

    print("[Method 5] CuPy cleanup complete")
except ImportError:
    print("[Method 5] CuPy not installed")
except Exception as e:
    print(f"[Method 5] Failed: {e}")

# Method 6: Check environment variables that might help
print("\n[Method 6] Checking CUDA environment variables...")
cuda_vars = {k: v for k, v in os.environ.items() if 'CUDA' in k or 'NVIDIA' in k}
if cuda_vars:
    for k, v in cuda_vars.items():
        print(f"  {k}={v}")
else:
    print("  No CUDA environment variables set")

# Method 7: Try numba if available
print("\n[Method 7] Attempting Numba CUDA cleanup...")
try:
    from numba import cuda

    for i in [0, 1]:
        try:
            cuda.select_device(i)
            cuda.close()
            print(f"  GPU {i}: Numba context closed")
        except Exception as e:
            print(f"  GPU {i}: Failed - {e}")

    print("[Method 7] Numba cleanup complete")
except ImportError:
    print("[Method 7] Numba not installed")
except Exception as e:
    print(f"[Method 7] Failed: {e}")

print("\n" + "=" * 80)
print("Cleanup attempts completed. Check nvidia-smi for results.")
print("=" * 80)
