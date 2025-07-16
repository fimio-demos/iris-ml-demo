#!/usr/bin/env python3
import time
import numpy as np
from numba import cuda

@cuda.jit
def big_kernel(a, b, out):
    i = cuda.grid(1)
    if i < a.size:
        out[i] = a[i] + b[i]

def main():
    n = 2_000_000_000  # ~8GB with float32

    print("Allocating large arrays...")
    a = np.ones(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    out = np.empty(n, dtype=np.float32)

    da = cuda.to_device(a)
    db = cuda.to_device(b)
    dout = cuda.device_array_like(out)

    print("Launching big_kernel...")
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    big_kernel[blocks_per_grid, threads_per_block](da, db, dout)
    cuda.synchronize()

    print("Sleeping for 5 minutes so you can check nvidia-smi...")
    time.sleep(300)

if __name__ == "__main__":
    main()
