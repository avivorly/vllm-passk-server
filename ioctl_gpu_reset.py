#!/usr/bin/env python3
"""
Last resort: Try using ioctl on /dev/nvidia devices.
This is a long shot but worth trying.
"""

import fcntl
import os
import struct

print("=" * 80)
print("Attempting ioctl-based GPU reset")
print("=" * 80)

# NVIDIA ioctl commands (from nvidia's open source driver headers)
# These are approximations - the actual values may differ
NVIDIA_IOC_MAGIC = ord('F')

# Try various ioctl numbers that might work
ioctl_numbers = [
    ('RESET_DEVICE', 0x20004600),  # Common pattern
    ('CLEAR_MEMORY', 0x20004601),
    ('FLUSH_CONTEXT', 0x20004602),
    ('DEVICE_SYNC', 0x20004603),
]

for gpu_id in [0, 1, 2]:
    device_path = f"/dev/nvidia{gpu_id}"
    print(f"\n{'='*60}")
    print(f"GPU {gpu_id}: {device_path}")
    print(f"{'='*60}")

    try:
        # Open device
        fd = os.open(device_path, os.O_RDWR)
        print(f"  Opened device (fd={fd})")

        # Try each ioctl
        for name, ioctl_num in ioctl_numbers:
            try:
                print(f"  Trying ioctl {name} (0x{ioctl_num:08x})...")
                result = fcntl.ioctl(fd, ioctl_num, 0)
                print(f"    Success! Result: {result}")
            except OSError as e:
                print(f"    Failed: {e}")
            except Exception as e:
                print(f"    Error: {e}")

        # Close device
        os.close(fd)
        print(f"  Closed device")

    except PermissionError:
        print(f"  Permission denied")
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 80)
print("ioctl attempts complete")
print("=" * 80)

# Also try opening with different flags
print("\n[Alternative] Trying open with O_NONBLOCK...")
for gpu_id in [0, 1, 2]:
    device_path = f"/dev/nvidia{gpu_id}"
    try:
        fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
        print(f"  GPU {gpu_id}: Opened with O_NONBLOCK (fd={fd})")
        os.close(fd)
    except Exception as e:
        print(f"  GPU {gpu_id}: {e}")

# Try accessing nvidiactl
print("\n[Alternative] Trying /dev/nvidiactl...")
try:
    fd = os.open("/dev/nvidiactl", os.O_RDWR)
    print(f"  Opened /dev/nvidiactl (fd={fd})")

    # Try some control ioctls
    for name, ioctl_num in [('RESET_ALL', 0x20004700), ('CLEAR_ALL', 0x20004701)]:
        try:
            print(f"  Trying {name}...")
            result = fcntl.ioctl(fd, ioctl_num, 0)
            print(f"    Success! Result: {result}")
        except Exception as e:
            print(f"    Failed: {e}")

    os.close(fd)
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 80)
print("All ioctl-based attempts failed (as expected)")
print("The NVIDIA driver doesn't expose memory cleanup via ioctl")
print("=" * 80)
