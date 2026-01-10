// scan_lookback_single_thread.cuh

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
__global__ void ScanDecoupledLookbackSingleThreadKernel(
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

    int tile_idx = s_tile_idx; // make local copy
    int gid = tile_idx * blockDim.x + threadIdx.x;

    // Step 2: Load and scan block
    int value = (gid < n) ? input[gid] : 0;
    value = BlockScan<BLOCK_SIZE>(value);

    // Step 3: Single thread does the decoupled lookback
    if (threadIdx.x == BLOCK_SIZE - 1) {
        // Publish block aggregate first
        TileDescriptor my_info;
        my_info.value = value;

        if (tile_idx == 0) {
            my_info.status = TileStatus::PREFIX;
        } else {
            my_info.status = TileStatus::AGGREGATE;
        }

        atomicExch(&tile_descriptors[tile_idx].raw, my_info.raw);
        __threadfence();

        if (tile_idx > 0) {
            int lookback_idx = tile_idx - 1;
            int running_prefix = 0;

            // Main lookback loop
            while (lookback_idx >= 0) {
                TileDescriptor pred_info; // predecessor info

                // Spin-wait for valid tile data
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

template<int BLOCK_SIZE>
void ScanDecoupledLookbackSingleThread(int* d_input, int* d_output, int n)
{
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    TileDescriptor* d_tile_descriptors;
    int* d_tile_counter;

    CUDA_CHECK(cudaMalloc(&d_tile_descriptors, num_blocks * sizeof(TileDescriptor)));
    CUDA_CHECK(cudaMalloc(&d_tile_counter, sizeof(int)));

    CUDA_CHECK(cudaMemset(d_tile_descriptors, 0, num_blocks * sizeof(TileDescriptor)));
    CUDA_CHECK(cudaMemset(d_tile_counter, 0, sizeof(int)));

    ScanDecoupledLookbackSingleThreadKernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
        d_input, d_output, n, d_tile_descriptors, d_tile_counter);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_tile_descriptors));
    CUDA_CHECK(cudaFree(d_tile_counter));
}