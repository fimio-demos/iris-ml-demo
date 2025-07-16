#!/usr/bin/env python3
import time
import numpy as np
from numba import cuda

@cuda.jit
def spike_kernel(a, b, out):
    i = cuda.grid(1)
    if i < a.size:
        out[i] = a[i] + b[i]

def main():
    n_big = 1_000_000_000  # ~4GB
    n_small = 500_000      # ~2MB

    for i in range(10):
        if i % 2 == 0:
            n = n_big
            print(f"Iteration {i+1}: Allocating big arrays (~4GB)")
        else:
            n = n_small
            print(f"Iteration {i+1}: Allocating small arrays (~2MB)")

        a = np.ones(n, dtype=np.float32)
        b = np.ones(n, dtype=np.float32)
        out = np.empty(n, dtype=np.float32)

        da = cuda.to_device(a)
        db = cuda.to_device(b)
        dout = cuda.device_array_like(out)

        threads_per_block = 256
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
        spike_kernel[blocks_per_grid, threads_per_block](da, db, dout)
        cuda.synchronize()

        print(f"Completed iteration {i+1}. Sleeping 10 seconds...")
        time.sleep(10)

    print("Done spiking GPU memory. Sleeping 5 min for final nvidia-smi check...")
    time.sleep(300)

if __name__ == "__main__":
    main()
