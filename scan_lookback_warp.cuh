// scan_lookback_warp.cuh
#pragma once

enum class TileStatus: int {
    INVALID = 0,
    AGGREGATE = 1,
    PREFIX = 2
};

union TileDescriptor {
    unsigned long long int raw;
    struct {
        int value;
        TileStatus status;
    };
};

template <int BLOCK_SIZE>
__global__ void ScanDecoupledLookbackWarpKernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    int n,
    TileDescriptor* tile_descriptors,
    int* g_tile_counter)
{
    __shared__ int s_tile_idx;
    __shared__ int s_prefix;

    // Step 1: Dynamically claim a tile index
    if (threadIdx.x == 0) {
        s_tile_idx = atomicAdd(g_tile_counter, 1);
    }
    __syncthreads();

    int tile_idx = s_tile_idx;
    int gid = tile_idx * BLOCK_SIZE + threadIdx.x;

    // Step 2: Load and compute block-local inclusive scan
    int value = (gid < n) ? input[gid] : 0;
    value = BlockScan<BLOCK_SIZE>(value);

    // Step 3: Last warp performs decoupled lookback
    int warp_idx = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    constexpr int last_warp_idx = BLOCK_SIZE / warpSize - 1;

    if (warp_idx == last_warp_idx) {
        // Broadcast block aggregate from last thread
        int block_aggregate = __shfl_sync(0xFFFFFFFF, value, warpSize - 1);

        // Publish our aggregate (or prefix if tile 0)
        if (lane == warpSize - 1) {
            TileDescriptor my_info;
            my_info.value = block_aggregate;
            my_info.status = (tile_idx > 0) ? TileStatus::AGGREGATE : TileStatus::PREFIX;
            atomicExch(&tile_descriptors[tile_idx].raw, my_info.raw);
            __threadfence();
        }
        __syncwarp();

        if (tile_idx == 0) {
            // First tile has no predecessor
            s_prefix = 0;
        } else {
            // Lookback through predecessors to compute exclusive prefix
            int lookback_base = tile_idx - 1;
            int running_prefix = 0;

            while (true) {
                // Each lane looks at a different predecessor tile
                int my_lookback_idx = lookback_base - lane;

                // Set defaults for out-of-bounds lanes (treated as PREFIX with value 0)
                TileDescriptor pred_info;
                pred_info.value = 0;
                pred_info.status = TileStatus::PREFIX;

                // In-bounds lanes spin until predecessor is valid
                if (my_lookback_idx >= 0) {
                    do {
                        pred_info.raw = atomicAdd(&tile_descriptors[my_lookback_idx].raw, 0);
                    } while (pred_info.status == TileStatus::INVALID);
                }

                // Find which lanes have a PREFIX (stopping point)
                unsigned prefix_mask = __ballot_sync(0xFFFFFFFF, pred_info.status == TileStatus::PREFIX);
                int prefix_lane = __ffs(prefix_mask) - 1;

                // Only sum contributions from lanes 0 through prefix_lane
                int my_contribution = (lane <= prefix_lane) ? pred_info.value : 0;

                // Warp-wide reduction to sum all contributions
                int iteration_sum = my_contribution;
                for (int i = 16; i >= 1; i /= 2) {
                    iteration_sum += __shfl_xor_sync(0xFFFFFFFF, iteration_sum, i);
                }
                running_prefix += iteration_sum;

                // If we found any PREFIX, we're done
                if (prefix_lane >= 0) {
                    break;
                }

                // All 32 were AGGREGATE, continue to earlier tiles
                lookback_base -= warpSize;
            }

            // Store prefix for other threads
            if (lane == 0) {
                s_prefix = running_prefix;
            }

            // Upgrade our status to PREFIX for later tiles
            if (lane == warpSize - 1) {
                TileDescriptor my_info;
                my_info.value = running_prefix + block_aggregate;
                my_info.status = TileStatus::PREFIX;
                atomicExch(&tile_descriptors[tile_idx].raw, my_info.raw);
                __threadfence();
            }
        }
    }
    __syncthreads();

    // Step 4: All threads add prefix and write final result
    if (gid < n) {
        output[gid] = s_prefix + value;
    }
}

template<int BLOCK_SIZE>
void ScanDecoupledLookbackWarp(int* d_input, int* d_output, int n)
{
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    TileDescriptor* d_tile_descriptors;
    int* d_tile_counter;

    CUDA_CHECK(cudaMalloc(&d_tile_descriptors, num_blocks * sizeof(TileDescriptor)));
    CUDA_CHECK(cudaMalloc(&d_tile_counter, sizeof(int)));

    CUDA_CHECK(cudaMemset(d_tile_descriptors, 0, num_blocks * sizeof(TileDescriptor)));
    CUDA_CHECK(cudaMemset(d_tile_counter, 0, sizeof(int)));

    ScanDecoupledLookbackWarpKernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
        d_input, d_output, n, d_tile_descriptors, d_tile_counter);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_tile_descriptors));
    CUDA_CHECK(cudaFree(d_tile_counter));
}