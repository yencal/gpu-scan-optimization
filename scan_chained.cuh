// scan_chained.cuh
#pragma once

template<int BLOCK_SIZE>
__global__ void ScanChainedKernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    int n,
    int* block_prefixes,
    int* block_status,
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

    // Step 3: Single thread waits for previous block and fetches prefix
    if (threadIdx.x == BLOCK_SIZE - 1) {
        if (logical_idx == 0) {
            s_prefix = 0;
        } else {
            // Spin-wait for previous block
            while (atomicAdd(&block_status[logical_idx - 1], 0) == 0) {
                // Spin
            }
            s_prefix = block_prefixes[logical_idx - 1];
        }

        // Publish: previous prefix + our block total
        block_prefixes[logical_idx] = s_prefix + value;
        __threadfence();  // Ensure prefix visible BEFORE flag
        atomicExch(&block_status[logical_idx], 1);
    }
    __syncthreads();

    // Step 4: All threads add prefix and write
    if (gid < n) {
        output[gid] = s_prefix + value;
    }
}

template<int BLOCK_SIZE>
void ScanChained(int* d_input, int* d_output, int n)
{
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int* d_block_prefixes;
    int* d_block_status;
    int* d_block_counter;

    CHECK_CUDA(cudaMalloc(&d_block_prefixes, num_blocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_block_status, num_blocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_block_counter, sizeof(int)));

    CHECK_CUDA(cudaMemset(d_block_status, 0, num_blocks * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_block_counter, 0, sizeof(int)));

    ScanChainedKernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(
        d_input, d_output, n, d_block_prefixes, d_block_status, d_block_counter);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(d_block_prefixes));
    CHECK_CUDA(cudaFree(d_block_status));
    CHECK_CUDA(cudaFree(d_block_counter));
}