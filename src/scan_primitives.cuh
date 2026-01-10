// scan_primitives.cuh
// Warp and block level scan primitives (inclusive)

#pragma once

#include <cuda_runtime.h>

// ============================================================================
// WARP SCAN (INCLUSIVE)
// ============================================================================
// Uses warp shuffle intrinsics for efficient intra-warp communication.
// No shared memory required.

static __device__ __forceinline__ int WarpScanInclusive(int value)
{
    const int lane = threadIdx.x % warpSize;
    
    #pragma unroll
    for (int offset = 1; offset < warpSize; offset *= 2) {
        int tmp = __shfl_up_sync(0xFFFFFFFF, value, offset);
        if (lane >= offset) {
            value += tmp;
        }
    }
    return value;
}

// ============================================================================
// BLOCK SCAN (INCLUSIVE)
// ============================================================================
// Two-level scan: warp scan within warps, then scan warp totals.

template<int BLOCK_SIZE>
static __device__ __forceinline__ int BlockScanInclusive(int value)
{
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be multiple of warp size");
    
    const int warp_idx = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;

    // Step 1: Warp-level inclusive scan
    int warp_scan = WarpScanInclusive(value);

    // Step 2: Last lane of each warp writes its total to shared memory
    __shared__ int warp_totals[NUM_WARPS];
    if (lane == warpSize - 1) {
        warp_totals[warp_idx] = warp_scan;
    }
    __syncthreads();

    // Step 3: First warp scans the warp totals
    if (warp_idx == 0) {
        int warp_total = (lane < NUM_WARPS) ? warp_totals[lane] : 0;
        warp_total = WarpScanInclusive(warp_total);
        if (lane < NUM_WARPS) {
            warp_totals[lane] = warp_total;
        }
    }
    __syncthreads();

    // Step 4: Add prefix from previous warps
    int warp_prefix = (warp_idx > 0) ? warp_totals[warp_idx - 1] : 0;
    return warp_scan + warp_prefix;
}

// ============================================================================
// BLOCK SCAN (EXCLUSIVE)
// ============================================================================
// Returns exclusive prefix sum: output[i] = sum of input[0..i-1]

template<int BLOCK_SIZE>
static __device__ __forceinline__ int BlockScanExclusive(int value)
{
    int inclusive = BlockScanInclusive<BLOCK_SIZE>(value);
    return inclusive - value;
}