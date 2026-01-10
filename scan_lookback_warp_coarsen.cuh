// scan_lookback_warp_coarsen.cuh
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

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void ScanDecoupledLookbackWarpCoarsenedKernel(
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

    int tile_idx = s_tile_idx;
    int tile_offset = tile_idx * TILE_SIZE;

    // Step 2: Load ITEMS_PER_THREAD elements per thread
    int items[ITEMS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = tile_offset + threadIdx.x + i * BLOCK_SIZE;
        items[i] = (idx < n) ? input[idx] : 0;
    }

    // Step 3: Thread-local inclusive scan
    #pragma unroll
    for (int i = 1; i < ITEMS_PER_THREAD; i++) {
        items[i] += items[i - 1];
    }

    // Step 4: BlockScan on thread totals (last item holds thread's sum)
    int thread_total = items[ITEMS_PER_THREAD - 1];
    int thread_prefix_exclusive = BlockScanExclusive<BLOCK_SIZE>(thread_total);

    // Step 5: Add thread prefix to all items
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items[i] += thread_prefix_exclusive;
    }

    // Now items[] holds block-local inclusive scan
    // Block aggregate is last thread's last item
    int block_aggregate = __shfl_sync(0xFFFFFFFF, items[ITEMS_PER_THREAD - 1], BLOCK_SIZE - 1);

    // Step 6: Warp lookback (same as before, just with larger tiles)
    int warp_idx = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    constexpr int last_warp_idx = BLOCK_SIZE / warpSize - 1;

    if (warp_idx == last_warp_idx) {
        // Publish aggregate
        if (lane == warpSize - 1) {
            TileDescriptor my_info;
            my_info.value = block_aggregate;
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
                int my_lookback_idx = lookback_base - lane;

                TileDescriptor pred_info;
                pred_info.value = 0;
                pred_info.status = TileStatus::PREFIX;

                if (my_lookback_idx >= 0) {
                    do {
                        pred_info.raw = atomicAdd(&tile_descriptors[my_lookback_idx].raw, 0);
                    } while (pred_info.status == TileStatus::INVALID);
                }

                unsigned prefix_mask = __ballot_sync(0xFFFFFFFF, pred_info.status == TileStatus::PREFIX);
                int prefix_lane = __ffs(prefix_mask) - 1;

                int my_contribution = (lane <= prefix_lane) ? pred_info.value : 0;

                for (int offset = 1; offset < warpSize; offset *= 2) {
                    int other = __shfl_up_sync(0xFFFFFFFF, my_contribution, offset);
                    if (lane >= offset) {
                        my_contribution += other;
                    }
                }

                int iteration_sum = __shfl_sync(0xFFFFFFFF, my_contribution, prefix_lane);
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
                my_info.value = exclusive_prefix + block_aggregate;
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
        int idx = tile_offset + threadIdx.x + i * BLOCK_SIZE;
        if (idx < n) {
            output[idx] = s_prefix + items[i];
        }
    }
}

template<int BLOCK_SIZE>
void ScanDecoupledLookbackWarpCoarsen(int* d_input, int* d_output, int n)
{
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    TileDescriptor* d_tile_descriptors;
    int* d_tile_counter;

    CUDA_CHECK(cudaMalloc(&d_tile_descriptors, num_blocks * sizeof(TileDescriptor)));
    CUDA_CHECK(cudaMalloc(&d_tile_counter, sizeof(int)));

    CUDA_CHECK(cudaMemset(d_tile_descriptors, 0, num_blocks * sizeof(TileDescriptor)));
    CUDA_CHECK(cudaMemset(d_tile_counter, 0, sizeof(int)));

    ScanDecoupledLookbackWarpCoarsenKernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
        d_input, d_output, n, d_tile_descriptors, d_tile_counter);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_tile_descriptors));
    CUDA_CHECK(cudaFree(d_tile_counter));
}