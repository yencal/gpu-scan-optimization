// scan_chained.cuh
// Chained scan: single-pass but serialized inter-tile communication
// 
// This implementation exists to demonstrate WHY decoupled lookback is needed.
// Each tile must wait for the previous tile to complete before it can finish,
// resulting in serialized execution despite being a "single kernel."

#pragma once

#include <cuda_runtime.h>
#include "utils.cuh"
#include "scan_primitives.cuh"

// ============================================================================
// KERNEL: CHAINED SCAN
// ============================================================================
// Algorithm:
// 1. Atomically claim a logical tile index (decouples hardware scheduling)
// 2. Load and scan tile locally
// 3. Wait for previous tile to publish its prefix (spin-wait)
// 4. Add prefix, write output, publish own prefix

template<int BLOCK_SIZE>
__global__ void ScanChainedKernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    int n,
    int* tile_prefixes,
    int* tile_ready,
    int* g_tile_counter)
{
    __shared__ int s_tile_idx;
    __shared__ int s_prefix;

    // Step 1: Dynamically claim logical tile index
    // This decouples logical order from hardware block scheduling
    if (threadIdx.x == 0) {
        s_tile_idx = atomicAdd(g_tile_counter, 1);
    }
    __syncthreads();

    const int tile_idx = s_tile_idx;
    const int gid = tile_idx * BLOCK_SIZE + threadIdx.x;

    // Step 2: Load and scan tile
    int value = (gid < n) ? input[gid] : 0;
    value = BlockScanInclusive<BLOCK_SIZE>(value);

    // Step 3: Single thread handles inter-tile communication
    if (threadIdx.x == BLOCK_SIZE - 1) {
        if (tile_idx == 0) {
            // First tile: no predecessor to wait for
            s_prefix = 0;
        } else {
            // Spin-wait for previous tile to be ready
            while (atomicAdd(&tile_ready[tile_idx - 1], 0) == 0) {
                // Spin
            }
            s_prefix = tile_prefixes[tile_idx - 1];
        }

        // Publish our prefix (previous prefix + our tile total)
        tile_prefixes[tile_idx] = s_prefix + value;
        __threadfence();  // Ensure prefix visible BEFORE ready flag
        atomicExch(&tile_ready[tile_idx], 1);
    }
    __syncthreads();

    // Step 4: All threads add prefix and write output
    if (gid < n) {
        output[gid] = s_prefix + value;
    }
}

// ============================================================================
// BENCHMARK WRAPPER
// ============================================================================

template<int BLOCK_SIZE>
struct ScanChained {
    static size_t GetTempSize(int n) {
        const int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return 2 * num_tiles * sizeof(int) + sizeof(int);
    }

    static void Run(int* d_input, int* d_output, int n, void* d_temp) {
        const int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Reset state (required before each run)
        CHECK_CUDA(cudaMemset(d_temp, 0, GetTempSize(n)));

        // Carve out temp buffer
        int* d_tile_prefixes = static_cast<int*>(d_temp);
        int* d_tile_ready = d_tile_prefixes + num_tiles;
        int* d_tile_counter = d_tile_ready + num_tiles;

        ScanChainedKernel<BLOCK_SIZE><<<num_tiles, BLOCK_SIZE>>>(
            d_input, d_output, n, d_tile_prefixes, d_tile_ready, d_tile_counter);
        CHECK_CUDA(cudaGetLastError());
    }
};
