#pragma once

enum BlockStatus {
    INVALID = 0,
    AGGREGATE = 1,
    PREFIX = 2
};

union TileDescriptor {
    unsigned long long int raw;
    struct {
        int value;
        int status;
    };
};

template<int BLOCK_SIZE>
__global__ void ScanDecoupledLookbackKernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    int n,
    TileDescriptor* tile_descriptors,
    int* g_block_counter)
{
    __shared__ int s_logical_idx;
    __shared__ int s_prefix;

    // Step 1: Dynamically claim logical block index
    if (threadIdx.x == 0) {
        s_logical_idx = atomicAdd(g_block_counter, 1);
    }
    __syncthreads();

    int logical_idx = s_logical_idx;
    int gid = logical_idx * BLOCK_SIZE + threadIdx.x;

    // Step 2: Load and scan block
    int value = (gid < n) ? input[gid] : 0;
    value = BlockScan<BLOCK_SIZE>(value);

    // Step 3: Single thread does the decoupled lookback
    if (threadIdx.x == BLOCK_SIZE - 1) {
        // Publish block aggregate first
        TileDescriptor my_info;
        my_info.value = value;

        if (logical_idx == 0) {
            my_info.status = PREFIX;
            s_prefix = 0;
        } else {
            my_info.status = AGGREGATE;
        }

        atomicExch(&tile_descriptors[logical_idx].raw, my_info.raw);
        __threadfence();

        if (logical_idx > 0) {
            int lookback_idx = logical_idx - 1;
            int running_prefix = 0;

            // Main lookback loop
            while (lookback_idx >= 0) {
                TileDescriptor pred_info;

                // Spin-wait for valid block data
                do {
                    pred_info.raw = atomicAdd(&tile_descriptors[lookback_idx].raw, 0);
                } while (pred_info.status == INVALID);

                running_prefix += pred_info.value;

                if (pred_info.status == PREFIX) {
                    break;
                }
                lookback_idx--;
            }

            s_prefix = running_prefix;

            // Upgrade to PREFIX
            my_info.value = s_prefix + value;
            my_info.status = PREFIX;
            atomicExch(&tile_descriptors[logical_idx].raw, my_info.raw);
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
void ScanDecoupledLookback(int* d_input, int* d_output, int n)
{
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    TileDescriptor* d_tile_descriptors;
    int* d_block_counter;

    CHECK_CUDA(cudaMalloc(&d_tile_descriptors, num_blocks * sizeof(TileDescriptor)));
    CHECK_CUDA(cudaMalloc(&d_block_counter, sizeof(int)));

    CHECK_CUDA(cudaMemset(d_tile_descriptors, 0, num_blocks * sizeof(TileDescriptor)));
    CHECK_CUDA(cudaMemset(d_block_counter, 0, sizeof(int)));

    ScanDecoupledLookbackKernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
        d_input, d_output, n, d_tile_descriptors, d_block_counter);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(d_tile_descriptors));
    CHECK_CUDA(cudaFree(d_block_counter));
}