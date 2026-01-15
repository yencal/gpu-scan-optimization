# GPU Scan Optimization

CUDA implementations of parallel prefix sum (scan), progressing from multi-kernel to single-pass decoupled lookback.

Companion code for [GPU Prefix Sum: From Multi-Kernel to Single-Pass Decoupled Lookback](https://yencal.github.io/gpu-prefix-sum-decoupled-lookback/).

## Results (H100)

| Algorithm | Time (ms) | Bandwidth | % Peak |
|-----------|-----------|-----------|--------|
| Multi-kernel (SMEM) | 10.72 | 801.52 GB/s | 23.9% |
| Multi-kernel (Warp shuffle) | 9.33 | 920.52 GB/s | 27.5% |
| Chained scan | 1767.29 | 4.86 GB/s | 0.14% |
| Lookback (single-thread) | 34.34 | 250.17 GB/s | 7.5% |
| Lookback (warp) | 15.06 | 570.55 GB/s | 17.0% |
| Lookback (warp + vectorized x4) | 5.29 | 1624.18 GB/s | 48.4% |
| Lookback (warp + vectorized x8) | 4.75 | 1807.59 GB/s | 53.9% |
| Lookback (warp + vectorized x12) | 3.44 | 2495.06 GB/s | 74.4% |
| CUB DeviceScan | 3.47 | 2478.55 GB/s | 73.9% |

## Build & Run
```bash
nvcc -O3 -std=c++17 -arch=sm_90 main.cu -o scan_bench
./scan_bench [power]  # default: 2^30 elements
```

## Files

- `scan_multi_kernel.cuh`: Three-phase reduce-then-scan
- `scan_chained.cuh`: Single-pass with serialized tile communication
- `scan_lookback.cuh`: Decoupled lookback (single-thread, warp, coarsened, vectorized)
- `scan_primitives.cuh`: Warp/block scan building blocks
- `tile_descriptor.cuh`: Tile status for lookback coordination
- `scan_cub.cuh`: CUB DeviceScan wrapper for comparison
- `utils.cuh`: Error checking and benchmark utilities