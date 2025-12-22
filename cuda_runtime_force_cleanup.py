#!/usr/bin/env python3
"""
Use CUDA Runtime API to try forcing memory cleanup.
This attempts to allocate ALL available memory to force eviction of zombie allocations.
"""

import ctypes
import sys

print("=" * 80)
print("CUDA Runtime API Force Memory Cleanup")
print("=" * 80)

try:
    # Load CUDA runtime library
    cuda_paths = [
        'libcudart.so',
        '/usr/local/cuda/lib64/libcudart.so',
        '/usr/lib/x86_64-linux-gnu/libcudart.so',
    ]

    cudart = None
    for path in cuda_paths:
        try:
            cudart = ctypes.CDLL(path)
            print(f"Loaded CUDA runtime from: {path}")
            break
        except:
            continue

    if not cudart:
        print("Could not load CUDA runtime library")
        sys.exit(1)

    # Define error codes
    CUDA_SUCCESS = 0

    for gpu_id in [0, 1]:
        print(f"\n{'='*60}")
        print(f"GPU {gpu_id}")
        print(f"{'='*60}")

        # Set device
        result = cudart.cudaSetDevice(gpu_id)
        if result != CUDA_SUCCESS:
            print(f"  cudaSetDevice failed: {result}")
            continue

        # Reset device first
        print("  Calling cudaDeviceReset()...")
        result = cudart.cudaDeviceReset()
        if result == CUDA_SUCCESS:
            print("    SUCCESS!")
        else:
            print(f"    Failed with code: {result}")

        # Get memory info
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        result = cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        if result == CUDA_SUCCESS:
            print(f"  Memory info:")
            print(f"    Free: {free.value / 1024**3:.2f} GB")
            print(f"    Total: {total.value / 1024**3:.2f} GB")
            print(f"    Used: {(total.value - free.value) / 1024**3:.2f} GB")

            # Try to allocate most of the free memory
            # This should force any "soft" allocations to be cleared
            alloc_size = int(free.value * 0.95)
            print(f"\n  Attempting to allocate {alloc_size / 1024**3:.2f} GB...")

            ptr = ctypes.c_void_p()
            result = cudart.cudaMalloc(ctypes.byref(ptr), alloc_size)

            if result == CUDA_SUCCESS:
                print(f"    Allocation successful at {hex(ptr.value)}")

                # Immediately free it
                print(f"    Freeing allocation...")
                result = cudart.cudaFree(ptr)
                if result == CUDA_SUCCESS:
                    print(f"    Free successful")
                else:
                    print(f"    Free failed: {result}")

                # Check memory again
                result = cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
                if result == CUDA_SUCCESS:
                    print(f"  Memory after alloc/free:")
                    print(f"    Free: {free.value / 1024**3:.2f} GB")
                    print(f"    Used: {(total.value - free.value) / 1024**3:.2f} GB")
            else:
                print(f"    Allocation failed: {result}")
                print(f"    (This is expected if zombie memory blocks allocation)")

        # Try device synchronize
        print(f"\n  Calling cudaDeviceSynchronize()...")
        result = cudart.cudaDeviceSynchronize()
        if result == CUDA_SUCCESS:
            print(f"    Success")
        else:
            print(f"    Failed: {result}")

        # Try to reset device again after operations
        print(f"\n  Calling cudaDeviceReset() again...")
        result = cudart.cudaDeviceReset()
        if result == CUDA_SUCCESS:
            print(f"    SUCCESS!")
        else:
            print(f"    Failed: {result}")

    print("\n" + "=" * 80)

    # Try torch as well for comparison
    print("\nAttempting PyTorch-based forced cleanup...")
    try:
        import torch

        for gpu_id in [0, 1]:
            print(f"\nGPU {gpu_id}:")
            with torch.cuda.device(gpu_id):
                # Try device reset via PyTorch
                try:
                    # This is not a standard PyTorch API but worth trying
                    torch.cuda.reset_peak_memory_stats(gpu_id)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(gpu_id)

                    # Get memory info
                    free, total = torch.cuda.mem_get_info(gpu_id)
                    print(f"  Memory (PyTorch):")
                    print(f"    Free: {free / 1024**3:.2f} GB")
                    print(f"    Total: {total / 1024**3:.2f} GB")

                except Exception as e:
                    print(f"  PyTorch cleanup error: {e}")

    except ImportError:
        print("PyTorch not available")

except Exception as e:
    print(f"Fatal error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Force cleanup complete. Check nvidia-smi.")
print("=" * 80)
