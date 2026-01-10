// scan_lookback.cuh
// Decoupled lookback scan implementations
//
// Key insight: Tiles don't wait for predecessors before scanning.
// They publish AGGREGATE immediately, then lookback to find PREFIX.
// This breaks the serialization in chained scan.
//
// Variants:
// - Single-thread lookback: one thread walks backward
// - Warp lookback: 32 threads check 32 predecessors in parallel
// - Coarsened: multiple elements per thread for larger tiles

#pragma once

#include <cuda_runtime.h>
#include "utils.cuh"
#include "scan_primitives.cuh"
#include "tile_descriptor.cuh"

// ============================================================================
// KERNEL: DECOUPLED LOOKBACK (SINGLE THREAD)
// ============================================================================
// One thread per tile does the lookback. Simpler but slower for deep chains.

template<int BLOCK_SIZE>
__global__ void ScanLookbackSingleThreadKernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    int n,
    TileDescriptor* tile_descriptors,
    int* g_tile_counter)
{
    __shared__ int s_tile_idx;
    __shared__ int s_prefix;

    // Step 1: Claim tile index
    if (threadIdx.x == 0) {
        s_tile_idx = atomicAdd(g_tile_counter, 1);
    }
    __syncthreads();

    const int tile_idx = s_tile_idx;
    const int gid = tile_idx * BLOCK_SIZE + threadIdx.x;

    // Step 2: Load and scan tile
    int value = (gid < n) ? input[gid] : 0;
    value = BlockScanInclusive<BLOCK_SIZE>(value);

    // Step 3: Single thread does the decoupled lookback
    if (threadIdx.x == BLOCK_SIZE - 1) {
        // Publish aggregate immediately (don't wait!)
        TileDescriptor my_info;
        my_info.value = value;
        my_info.status = (tile_idx == 0) ? TileStatus::PREFIX : TileStatus::AGGREGATE;
        atomicExch(&tile_descriptors[tile_idx].raw, my_info.raw);
        __threadfence();

        if (tile_idx == 0) {
            s_prefix = 0;
        } else {
            int lookback_idx = tile_idx - 1;
            int running_prefix = 0;

            // Lookback loop: walk backward until we find PREFIX
            while (lookback_idx >= 0) {
                TileDescriptor pred_info;

                // Spin-wait for valid data
                do {
                    pred_info.raw = atomicAdd(&tile_descriptors[lookback_idx].raw, 0);
                } while (pred_info.status == TileStatus::INVALID);

                running_prefix += pred_info.value;

                if (pred_info.status == TileStatus::PREFIX) {
                    break;
                }
                lookback_idx--;
            }

            s_prefix = running_prefix;

            // Upgrade to PREFIX
            my_info.value = s_prefix + value;
            my_info.status = TileStatus::PREFIX;
            atomicExch(&tile_descriptors[tile_idx].raw, my_info.raw);
            __threadfence();
        }
    }
    __syncthreads();

    // Step 4: All threads add prefix and write
    if (gid < n) {
        output[gid] = s_prefix + value;
    }
}

// ============================================================================
// KERNEL: DECOUPLED LOOKBACK (WARP)
// ============================================================================
// Entire warp cooperates on lookback: 32 predecessors checked in parallel.
// Much faster for deep lookback chains.

template<int BLOCK_SIZE>
__global__ void ScanLookbackWarpKernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    int n,
    TileDescriptor* tile_descriptors,
    int* g_tile_counter)
{
    __shared__ int s_tile_idx;
    __shared__ int s_prefix;

    // Step 1: Claim tile index
    if (threadIdx.x == 0) {
        s_tile_idx = atomicAdd(g_tile_counter, 1);
    }
    __syncthreads();

    const int tile_idx = s_tile_idx;
    const int gid = tile_idx * BLOCK_SIZE + threadIdx.x;

    // Step 2: Load and scan tile
    int value = (gid < n) ? input[gid] : 0;
    value = BlockScanInclusive<BLOCK_SIZE>(value);

    // Step 3: Last warp does the decoupled lookback
    const int warp_idx = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    constexpr int LAST_WARP = BLOCK_SIZE / warpSize - 1;

    if (warp_idx == LAST_WARP) {
        // Get tile aggregate from last thread
        const int tile_aggregate = __shfl_sync(0xFFFFFFFF, value, warpSize - 1);

        // Publish aggregate (only one thread writes)
        if (lane == warpSize - 1) {
            TileDescriptor my_info;
            my_info.value = tile_aggregate;
            my_info.status = (tile_idx == 0) ? TileStatus::PREFIX : TileStatus::AGGREGATE;
            atomicExch(&tile_descriptors[tile_idx].raw, my_info.raw);
            __threadfence();
        }
        __syncwarp();

        if (tile_idx == 0) {
            s_prefix = 0;
        } else {
            int exclusive_prefix = 0;
            int lookback_base = tile_idx - 1;

            while (true) {
                // Each lane checks a different predecessor
                const int my_lookback_idx = lookback_base - lane;

                TileDescriptor pred_info;
                pred_info.value = 0;
                pred_info.status = TileStatus::PREFIX;  // Default for out-of-bounds

                if (my_lookback_idx >= 0) {
                    do {
                        pred_info.raw = atomicAdd(&tile_descriptors[my_lookback_idx].raw, 0);
                    } while (pred_info.status == TileStatus::INVALID);
                }

                // Find which lanes found PREFIX
                const unsigned prefix_mask = __ballot_sync(0xFFFFFFFF, 
                    pred_info.status == TileStatus::PREFIX);
                const int prefix_lane = __ffs(prefix_mask) - 1;

                // Sum values from lane 0 through prefix_lane
                int contribution = (lane <= prefix_lane) ? pred_info.value : 0;

                // Warp inclusive scan to sum contributions
                #pragma unroll
                for (int offset = 1; offset < warpSize; offset *= 2) {
                    int tmp = __shfl_up_sync(0xFFFFFFFF, contribution, offset);
                    if (lane >= offset) {
                        contribution += tmp;
                    }
                }

                // Get sum from prefix_lane
                const int iteration_sum = __shfl_sync(0xFFFFFFFF, contribution, prefix_lane);
                exclusive_prefix += iteration_sum;

                // Done if we found PREFIX or reached beginning
                if (prefix_lane < warpSize - 1 || my_lookback_idx <= 0) {
                    break;
                }

                // All 32 were AGGREGATE, continue lookback
                lookback_base -= warpSize;
            }

            if (lane == 0) {
                s_prefix = exclusive_prefix;
            }

            // Upgrade to PREFIX
            if (lane == warpSize - 1) {
                TileDescriptor my_info;
                my_info.value = exclusive_prefix + tile_aggregate;
                my_info.status = TileStatus::PREFIX;
                atomicExch(&tile_descriptors[tile_idx].raw, my_info.raw);
                __threadfence();
            }
        }
    }
    __syncthreads();

    // Step 4: All threads add prefix and write
    if (gid < n) {
        output[gid] = s_prefix + value;
    }
}

// ============================================================================
// KERNEL: DECOUPLED LOOKBACK (WARP + COARSENED)
// ============================================================================
// Each thread processes ITEMS_PER_THREAD elements.
// Fewer tiles = shorter lookback chains.

template<int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void ScanLookbackWarpCoarsenedKernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    int n,
    TileDescriptor* tile_descriptors,
    int* g_tile_counter)
{
    constexpr int TILE_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;

    __shared__ int s_tile_idx;
    __shared__ int s_prefix;

    // Step 1: Claim tile index
    if (threadIdx.x == 0) {
        s_tile_idx = atomicAdd(g_tile_counter, 1);
    }
    __syncthreads();

    const int tile_idx = s_tile_idx;
    const int tile_offset = tile_idx * TILE_SIZE;

    // Step 2: Load ITEMS_PER_THREAD elements per thread
    int items[ITEMS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        const int idx = tile_offset + threadIdx.x + i * BLOCK_SIZE;
        items[i] = (idx < n) ? input[idx] : 0;
    }

    // Step 3: Thread-local inclusive scan
    #pragma unroll
    for (int i = 1; i < ITEMS_PER_THREAD; i++) {
        items[i] += items[i - 1];
    }

    // Step 4: BlockScan on thread totals
    const int thread_total = items[ITEMS_PER_THREAD - 1];
    const int thread_prefix = BlockScanExclusive<BLOCK_SIZE>(thread_total);

    // Step 5: Add thread prefix to all items
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items[i] += thread_prefix;
    }

    // Tile aggregate is last thread's last item
    const int tile_aggregate = __shfl_sync(0xFFFFFFFF, 
        items[ITEMS_PER_THREAD - 1], BLOCK_SIZE - 1);

    // Step 6: Warp lookback
    const int warp_idx = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    constexpr int LAST_WARP = BLOCK_SIZE / warpSize - 1;

    if (warp_idx == LAST_WARP) {
        // Publish aggregate
        if (lane == warpSize - 1) {
            TileDescriptor my_info;
            my_info.value = tile_aggregate;
            my_info.status = (tile_idx == 0) ? TileStatus::PREFIX : TileStatus::AGGREGATE;
            atomicExch(&tile_descriptors[tile_idx].raw, my_info.raw);
            __threadfence();
        }
        __syncwarp();

        if (tile_idx == 0) {
            s_prefix = 0;
        } else {
            int exclusive_prefix = 0;
            int lookback_base = tile_idx - 1;

            while (true) {
                const int my_lookback_idx = lookback_base - lane;

                TileDescriptor pred_info;
                pred_info.value = 0;
                pred_info.status = TileStatus::PREFIX;

                if (my_lookback_idx >= 0) {
                    do {
                        pred_info.raw = atomicAdd(&tile_descriptors[my_lookback_idx].raw, 0);
                    } while (pred_info.status == TileStatus::INVALID);
                }

                const unsigned prefix_mask = __ballot_sync(0xFFFFFFFF, 
                    pred_info.status == TileStatus::PREFIX);
                const int prefix_lane = __ffs(prefix_mask) - 1;

                int contribution = (lane <= prefix_lane) ? pred_info.value : 0;

                #pragma unroll
                for (int offset = 1; offset < warpSize; offset *= 2) {
                    int tmp = __shfl_up_sync(0xFFFFFFFF, contribution, offset);
                    if (lane >= offset) {
                        contribution += tmp;
                    }
                }

                const int iteration_sum = __shfl_sync(0xFFFFFFFF, contribution, prefix_lane);
                exclusive_prefix += iteration_sum;

                if (prefix_lane < warpSize - 1 || my_lookback_idx <= 0) {
                    break;
                }

                lookback_base -= warpSize;
            }

            if (lane == 0) {
                s_prefix = exclusive_prefix;
            }

            if (lane == warpSize - 1) {
                TileDescriptor my_info;
                my_info.value = exclusive_prefix + tile_aggregate;
                my_info.status = TileStatus::PREFIX;
                atomicExch(&tile_descriptors[tile_idx].raw, my_info.raw);
                __threadfence();
            }
        }
    }
    __syncthreads();

    // Step 7: Add global prefix and write
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        const int idx = tile_offset + threadIdx.x + i * BLOCK_SIZE;
        if (idx < n) {
            output[idx] = s_prefix + items[i];
        }
    }
}

// ============================================================================
// BENCHMARK WRAPPERS
// ============================================================================

template<int BLOCK_SIZE>
struct ScanLookbackSingleThread {
    static size_t GetTempSize(int n) {
        const int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return num_tiles * sizeof(TileDescriptor) + sizeof(int);
    }

    static void Run(int* d_input, int* d_output, int n, void* d_temp) {
        const int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Reset state (required before each run)
        CHECK_CUDA(cudaMemset(d_temp, 0, GetTempSize(n)));

        // Carve out temp buffer
        TileDescriptor* d_tile_descriptors = static_cast<TileDescriptor*>(d_temp);
        int* d_tile_counter = reinterpret_cast<int*>(d_tile_descriptors + num_tiles);

        ScanLookbackSingleThreadKernel<BLOCK_SIZE><<<num_tiles, BLOCK_SIZE>>>(
            d_input, d_output, n, d_tile_descriptors, d_tile_counter);
        CHECK_CUDA(cudaGetLastError());
    }
};

template<int BLOCK_SIZE>
struct ScanLookbackWarp {
    static size_t GetTempSize(int n) {
        const int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        return num_tiles * sizeof(TileDescriptor) + sizeof(int);
    }

    static void Run(int* d_input, int* d_output, int n, void* d_temp) {
        const int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Reset state (required before each run)
        CHECK_CUDA(cudaMemset(d_temp, 0, GetTempSize(n)));

        // Carve out temp buffer
        TileDescriptor* d_tile_descriptors = static_cast<TileDescriptor*>(d_temp);
        int* d_tile_counter = reinterpret_cast<int*>(d_tile_descriptors + num_tiles);

        ScanLookbackWarpKernel<BLOCK_SIZE><<<num_tiles, BLOCK_SIZE>>>(
            d_input, d_output, n, d_tile_descriptors, d_tile_counter);
        CHECK_CUDA(cudaGetLastError());
    }
};

template<int BLOCK_SIZE, int ITEMS_PER_THREAD>
struct ScanLookbackWarpCoarsened {
    static constexpr int TILE_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;

    static size_t GetTempSize(int n) {
        const int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
        return num_tiles * sizeof(TileDescriptor) + sizeof(int);
    }

    static void Run(int* d_input, int* d_output, int n, void* d_temp) {
        const int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;

        // Reset state (required before each run)
        CHECK_CUDA(cudaMemset(d_temp, 0, GetTempSize(n)));

        // Carve out temp buffer
        TileDescriptor* d_tile_descriptors = static_cast<TileDescriptor*>(d_temp);
        int* d_tile_counter = reinterpret_cast<int*>(d_tile_descriptors + num_tiles);

        ScanLookbackWarpCoarsenedKernel<BLOCK_SIZE, ITEMS_PER_THREAD><<<num_tiles, BLOCK_SIZE>>>(
            d_input, d_output, n, d_tile_descriptors, d_tile_counter);
        CHECK_CUDA(cudaGetLastError());
    }
};
