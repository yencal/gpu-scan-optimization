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
// KERNEL: INITIALIZATION OF BUFFERS
// ============================================================================

template<int BLOCK_SIZE>
__global__ void InitTileState(TileDescriptor* tile_descriptors, int* tile_counter, int num_tiles) {
    for (int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; 
         idx < num_tiles; 
         idx += gridDim.x * BLOCK_SIZE) {
        tile_descriptors[idx].raw = 0;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *tile_counter = 0;
    }
}

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
    __shared__ int s_tile_aggregate;
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

    // Last thread writes aggregate to shared memory
    if (threadIdx.x == BLOCK_SIZE - 1) {
        s_tile_aggregate = value;
    }
    __syncthreads();

    // Step 3: Thread 0 does the decoupled lookback
    if (threadIdx.x == 0) {
        const int tile_aggregate = s_tile_aggregate;

        // Publish aggregate immediately (don't wait!)
        TileDescriptor my_info;
        my_info.value = tile_aggregate;
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
            my_info.value = running_prefix + tile_aggregate;
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
    __shared__ int s_tile_aggregate;
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

    // Last thread writes aggregate to shared memory
    if (threadIdx.x == BLOCK_SIZE - 1) {
        s_tile_aggregate = value;
    }
    __syncthreads();

    // Step 3: Warp 0 does the decoupled lookback
    const int warp_idx = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;

    if (warp_idx == 0) {
        const int tile_aggregate = s_tile_aggregate;

        // Publish aggregate (thread 0 writes)
        if (threadIdx.x == 0) {
            TileDescriptor my_info;
            my_info.value = tile_aggregate;
            my_info.status = (tile_idx == 0) ? TileStatus::PREFIX : TileStatus::AGGREGATE;
            atomicExch(&tile_descriptors[tile_idx].raw, my_info.raw);
            __threadfence();
        }
        __syncwarp();

        if (tile_idx == 0) {
            if (threadIdx.x == 0) {
                s_prefix = 0;
            }
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
                const int prefix_lane = __ffs(prefix_mask) - 1;  // -1 if none found

                // Include all lanes if no PREFIX found, otherwise lanes 0..prefix_lane
                int contribution = (prefix_lane < 0 || lane <= prefix_lane) ? pred_info.value : 0;

                // XOR reduction - all lanes get the sum
                #pragma unroll
                for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                    contribution += __shfl_xor_sync(0xFFFFFFFF, contribution, offset);
                }

                exclusive_prefix += contribution;

                // If we found any PREFIX, we're done
                if (prefix_lane >= 0) {
                    break;
                }

                // All 32 were AGGREGATE, continue to earlier tiles
                lookback_base -= warpSize;
            }

            // Thread 0 writes prefix to shared memory
            if (threadIdx.x == 0) {
                s_prefix = exclusive_prefix;

                // Upgrade to PREFIX
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
// Uses blocked layout (simple, better for local scan).

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
    __shared__ int s_tile_aggregate;
    __shared__ int s_prefix;

    // Step 1: Claim tile index
    if (threadIdx.x == 0) {
        s_tile_idx = atomicAdd(g_tile_counter, 1);
    }
    __syncthreads();

    const int tile_idx = s_tile_idx;
    const int tile_offset = tile_idx * TILE_SIZE;

    // Step 2: Load ITEMS_PER_THREAD elements per thread (blocked layout)
    int items[ITEMS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        const int idx = tile_offset + threadIdx.x * ITEMS_PER_THREAD + i;
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

    // Last thread writes tile aggregate to shared memory
    if (threadIdx.x == BLOCK_SIZE - 1) {
        s_tile_aggregate = items[ITEMS_PER_THREAD - 1];
    }
    __syncthreads();

    // Step 6: Warp 0 does the decoupled lookback
    const int warp_idx = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;

    if (warp_idx == 0) {
        const int tile_aggregate = s_tile_aggregate;

        // Publish aggregate (thread 0 writes)
        if (threadIdx.x == 0) {
            TileDescriptor my_info;
            my_info.value = tile_aggregate;
            my_info.status = (tile_idx == 0) ? TileStatus::PREFIX : TileStatus::AGGREGATE;
            atomicExch(&tile_descriptors[tile_idx].raw, my_info.raw);
            __threadfence();
        }
        __syncwarp();

        if (tile_idx == 0) {
            if (threadIdx.x == 0) {
                s_prefix = 0;
            }
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
                const int prefix_lane = __ffs(prefix_mask) - 1;  // -1 if none found

                // Include all lanes if no PREFIX found, otherwise lanes 0..prefix_lane
                int contribution = (prefix_lane < 0 || lane <= prefix_lane) ? pred_info.value : 0;

                // XOR reduction - all lanes get the sum
                #pragma unroll
                for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                    contribution += __shfl_xor_sync(0xFFFFFFFF, contribution, offset);
                }

                exclusive_prefix += contribution;

                // If we found any PREFIX, we're done
                if (prefix_lane >= 0) {
                    break;
                }

                // All 32 were AGGREGATE, continue to earlier tiles
                lookback_base -= warpSize;
            }

            // Thread 0 writes prefix to shared memory
            if (threadIdx.x == 0) {
                s_prefix = exclusive_prefix;

                // Upgrade to PREFIX
                TileDescriptor my_info;
                my_info.value = exclusive_prefix + tile_aggregate;
                my_info.status = TileStatus::PREFIX;
                atomicExch(&tile_descriptors[tile_idx].raw, my_info.raw);
                __threadfence();
            }
        }
    }
    __syncthreads();

    // Step 7: Add global prefix and write (blocked layout)
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        const int idx = tile_offset + threadIdx.x * ITEMS_PER_THREAD + i;
        if (idx < n) {
            output[idx] = s_prefix + items[i];
        }
    }
}

// ============================================================================
// KERNEL: DECOUPLED LOOKBACK (WARP + VECTORIZED)
// ============================================================================
// Uses int4 loads/stores for better memory throughput.
// VEC_LOADS = number of int4 loads per thread (ITEMS_PER_THREAD = VEC_LOADS * 4)

template<int BLOCK_SIZE, int VEC_LOADS>
__global__ void ScanLookbackWarpCoarsenedVectorizedKernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    int n,
    TileDescriptor* tile_descriptors,
    int* g_tile_counter)
{
    constexpr int ITEMS_PER_THREAD = VEC_LOADS * 4;
    constexpr int TILE_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;

    __shared__ int s_tile_idx;
    __shared__ int s_tile_aggregate;
    __shared__ int s_prefix;

    // Step 1: Claim tile index
    if (threadIdx.x == 0) {
        s_tile_idx = atomicAdd(g_tile_counter, 1);
    }
    __syncthreads();

    const int tile_idx = s_tile_idx;
    const int tile_offset = tile_idx * TILE_SIZE;

    // Step 2: Load using int4 (vectorized, blocked layout)
    int items[ITEMS_PER_THREAD];
    const int4* input_vec = reinterpret_cast<const int4*>(input + tile_offset);

    #pragma unroll
    for (int v = 0; v < VEC_LOADS; v++) {
        const int vec_idx = threadIdx.x * VEC_LOADS + v;
        const int base_idx = tile_offset + vec_idx * 4;

        if (base_idx + 3 < n) {
            int4 loaded = input_vec[vec_idx];
            items[v * 4 + 0] = loaded.x;
            items[v * 4 + 1] = loaded.y;
            items[v * 4 + 2] = loaded.z;
            items[v * 4 + 3] = loaded.w;
        } else {
            // Boundary handling - scalar fallback
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int idx = base_idx + i;
                items[v * 4 + i] = (idx < n) ? input[idx] : 0;
            }
        }
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

    // Last thread writes tile aggregate to shared memory
    if (threadIdx.x == BLOCK_SIZE - 1) {
        s_tile_aggregate = items[ITEMS_PER_THREAD - 1];
    }
    __syncthreads();

    // Step 6: Warp 0 does the decoupled lookback
    const int warp_idx = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;

    if (warp_idx == 0) {
        const int tile_aggregate = s_tile_aggregate;

        // Publish aggregate (thread 0 writes)
        if (threadIdx.x == 0) {
            TileDescriptor my_info;
            my_info.value = tile_aggregate;
            my_info.status = (tile_idx == 0) ? TileStatus::PREFIX : TileStatus::AGGREGATE;
            atomicExch(&tile_descriptors[tile_idx].raw, my_info.raw);
            __threadfence();
        }
        __syncwarp();

        if (tile_idx == 0) {
            if (threadIdx.x == 0) {
                s_prefix = 0;
            }
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
                const int prefix_lane = __ffs(prefix_mask) - 1;  // -1 if none found

                // Include all lanes if no PREFIX found, otherwise lanes 0..prefix_lane
                int contribution = (prefix_lane < 0 || lane <= prefix_lane) ? pred_info.value : 0;

                // XOR reduction - all lanes get the sum
                #pragma unroll
                for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                    contribution += __shfl_xor_sync(0xFFFFFFFF, contribution, offset);
                }

                exclusive_prefix += contribution;

                // If we found any PREFIX, we're done
                if (prefix_lane >= 0) {
                    break;
                }

                // All 32 were AGGREGATE, continue to earlier tiles
                lookback_base -= warpSize;
            }

            // Thread 0 writes prefix to shared memory
            if (threadIdx.x == 0) {
                s_prefix = exclusive_prefix;

                // Upgrade to PREFIX
                TileDescriptor my_info;
                my_info.value = exclusive_prefix + tile_aggregate;
                my_info.status = TileStatus::PREFIX;
                atomicExch(&tile_descriptors[tile_idx].raw, my_info.raw);
                __threadfence();
            }
        }
    }
    __syncthreads();

    // Step 7: Add global prefix and store using int4 (vectorized)
    int4* output_vec = reinterpret_cast<int4*>(output + tile_offset);

    #pragma unroll
    for (int v = 0; v < VEC_LOADS; v++) {
        const int vec_idx = threadIdx.x * VEC_LOADS + v;
        const int base_idx = tile_offset + vec_idx * 4;

        if (base_idx + 3 < n) {
            int4 result;
            result.x = s_prefix + items[v * 4 + 0];
            result.y = s_prefix + items[v * 4 + 1];
            result.z = s_prefix + items[v * 4 + 2];
            result.w = s_prefix + items[v * 4 + 3];
            output_vec[vec_idx] = result;
        } else {
            // Boundary handling - scalar fallback
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int idx = base_idx + i;
                if (idx < n) {
                    output[idx] = s_prefix + items[v * 4 + i];
                }
            }
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

        // Carve out temp buffer
        TileDescriptor* d_tile_descriptors = static_cast<TileDescriptor*>(d_temp);
        int* d_tile_counter = reinterpret_cast<int*>(d_tile_descriptors + num_tiles);

        // Init/Reset buffers (required before each run)
        const int max_grid_size = 1024;
        const int init_grid = min((num_tiles + BLOCK_SIZE - 1) / BLOCK_SIZE, max_grid_size);
        InitTileState<BLOCK_SIZE><<<init_grid, BLOCK_SIZE>>>(d_tile_descriptors, d_tile_counter, num_tiles);

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

        // Carve out temp buffer
        TileDescriptor* d_tile_descriptors = static_cast<TileDescriptor*>(d_temp);
        int* d_tile_counter = reinterpret_cast<int*>(d_tile_descriptors + num_tiles);

        // Init/Reset buffers (required before each run)
        const int max_grid_size = 1024;
        const int init_grid = min((num_tiles + BLOCK_SIZE - 1) / BLOCK_SIZE, max_grid_size);
        InitTileState<BLOCK_SIZE><<<init_grid, BLOCK_SIZE>>>(d_tile_descriptors, d_tile_counter, num_tiles);
                
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

        // Carve out temp buffer
        TileDescriptor* d_tile_descriptors = static_cast<TileDescriptor*>(d_temp);
        int* d_tile_counter = reinterpret_cast<int*>(d_tile_descriptors + num_tiles);

        // Init/Reset buffers (required before each run)
        const int max_grid_size = 1024;
        const int init_grid = min((num_tiles + BLOCK_SIZE - 1) / BLOCK_SIZE, max_grid_size);
        InitTileState<BLOCK_SIZE><<<init_grid, BLOCK_SIZE>>>(d_tile_descriptors, d_tile_counter, num_tiles);

        ScanLookbackWarpCoarsenedKernel<BLOCK_SIZE, ITEMS_PER_THREAD><<<num_tiles, BLOCK_SIZE>>>(
            d_input, d_output, n, d_tile_descriptors, d_tile_counter);
        CHECK_CUDA(cudaGetLastError());
    }
};

template<int BLOCK_SIZE, int VEC_LOADS>
struct ScanLookbackWarpCoarsenedVectorized {
    static constexpr int ITEMS_PER_THREAD = VEC_LOADS * 4;
    static constexpr int TILE_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;

    static size_t GetTempSize(int n) {
        const int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
        return num_tiles * sizeof(TileDescriptor) + sizeof(int);
    }

    static void Run(int* d_input, int* d_output, int n, void* d_temp) {
        const int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;

        // Carve out temp buffer
        TileDescriptor* d_tile_descriptors = static_cast<TileDescriptor*>(d_temp);
        int* d_tile_counter = reinterpret_cast<int*>(d_tile_descriptors + num_tiles);

        // Init/Reset buffers (required before each run)
        const int max_grid_size = 1024;
        const int init_grid = min((num_tiles + BLOCK_SIZE - 1) / BLOCK_SIZE, max_grid_size);
        InitTileState<BLOCK_SIZE><<<init_grid, BLOCK_SIZE>>>(d_tile_descriptors, d_tile_counter, num_tiles);

        ScanLookbackWarpCoarsenedVectorizedKernel<BLOCK_SIZE, VEC_LOADS><<<num_tiles, BLOCK_SIZE>>>(
            d_input, d_output, n, d_tile_descriptors, d_tile_counter);
        CHECK_CUDA(cudaGetLastError());
    }
};
