#!/usr/bin/env python3
import time
import numpy as np
from numba import cuda

# A simple vector add kernel
@cuda.jit
def vector_add(a, b, out):
    i = cuda.grid(1)
    if i < a.size:
        out[i] = a[i] + b[i]

def main():
    # size of our arrays
    n = 10_000_000

    # allocate host arrays
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32) * 2
    out = np.empty(n, dtype=np.float32)

    # copy to device
    da = cuda.to_device(a)
    db = cuda.to_device(b)
    dout = cuda.device_array_like(out)

    # launch kernel
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    print(f"Launching vector_add[{blocks_per_grid}, {threads_per_block}] on GPU…")
    vector_add[blocks_per_grid, threads_per_block](da, db, dout)
    cuda.synchronize()

    # copy result back and verify
    dout.copy_to_host(out)
    print("First 5 results:", out[:5])
    assert np.allclose(out, a + b)

    # keep process alive so you can inspect via nvidia-smi
    print("GPU work done—sleeping for 5 minutes to let you inspect nvidia-smi…")
    time.sleep(300)

if __name__ == "__main__":
    main()
