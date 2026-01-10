# GPU Scan Optimization

CUDA implementations of parallel prefix sum (scan), progressing from multi-kernel to single-pass decoupled lookback.

Companion code for [blog post title](link).

## Results (RTX 6000 Ada)

| Algorithm | Bandwidth | % Peak |
|-----------|-----------|--------|
| Multi-kernel (baseline) | 405 GB/s | 42% |
| Chained scan | 4.5 GB/s | 0.5% |
| Lookback (single-thread) | 262 GB/s | 27% |
| Lookback (warp) | 520 GB/s | 54% |
| Lookback (warp + coarsened x4) | 819 GB/s | 85% |

## Build & Run
```bash
cd src
nvcc -O3 -std=c++17 -arch=sm_89 main.cu -o scan_bench
./scan_bench [power]  # default: 2^28 elements
```

## Files

- `scan_multi_kernel.cuh` — Three-phase reduce-then-scan
- `scan_chained.cuh` — Single-pass with serialized tile communication
- `scan_lookback.cuh` — Decoupled lookback (single-thread, warp, coarsened)
- `scan_primitives.cuh` — Warp/block scan building blocks
- `tile_descriptor.cuh` — Tile status for lookback coordination