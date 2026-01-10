// scan_multi_kernel.cuh
// Multi-kernel scan implementations

#pragma once

#include <cuda_runtime.h>
#include "utils.cuh"
#include "scan_primitives.cuh"

// ============================================================================
// KERNEL: SCAN TILES (SHARED MEMORY BASELINE)
// ============================================================================
// Double-buffered shared memory scan. Included for comparison only.
// Warp shuffle version below is faster.

template<int BLOCK_SIZE>
__global__ void ScanTilesSMEM(
    const int* __restrict__ input,
    int* __restrict__ output,
    int n,
    int* tile_aggregates)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    __shared__ int smem[2][BLOCK_SIZE];

    int wbuf = 0;
    smem[wbuf][tid] = (gid < n) ? input[gid] : 0;
    __syncthreads();

    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        wbuf = 1 - wbuf;
        int rbuf = 1 - wbuf;
        if (tid >= offset) {
            smem[wbuf][tid] = smem[rbuf][tid - offset] + smem[rbuf][tid];
        } else {
            smem[wbuf][tid] = smem[rbuf][tid];
        }
        __syncthreads();
    }

    if (gid < n) {
        output[gid] = smem[wbuf][tid];
    }
    
    if (tid == BLOCK_SIZE - 1 && tile_aggregates != nullptr) {
        tile_aggregates[blockIdx.x] = smem[wbuf][BLOCK_SIZE - 1];
    }
}

// ============================================================================
// KERNEL: SCAN TILES (WARP SHUFFLE)
// ============================================================================
// Uses warp shuffle primitives for efficient block scan.

template<int BLOCK_SIZE>
__global__ void ScanTilesWarp(
    const int* __restrict__ input,
    int* __restrict__ output,
    int n,
    int* tile_aggregates)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int value = (gid < n) ? input[gid] : 0;
    value = BlockScanInclusive<BLOCK_SIZE>(value);

    if (gid < n) {
        output[gid] = value;
    }

    if (threadIdx.x == BLOCK_SIZE - 1 && tile_aggregates != nullptr) {
        tile_aggregates[blockIdx.x] = value;
    }
}

// ============================================================================
// KERNEL: ADD TILE PREFIXES
// ============================================================================
// Final pass: adds scanned tile prefix to each element.

__global__ void AddTilePrefixes(
    int* output,
    int n,
    const int* scanned_tile_aggregates)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0 && gid < n) {
        output[gid] += scanned_tile_aggregates[blockIdx.x - 1];
    }
}

// ============================================================================
// MULTI-KERNEL SCAN: SHARED MEMORY (RECURSIVE - EDUCATIONAL)
// ============================================================================
// Three-phase recursive algorithm:
// 1. Scan each tile independently, extract tile aggregates
// 2. Recursively scan tile aggregates
// 3. Add scanned prefixes back to each tile
//
// This version allocates memory in each recursive call for clarity.
// See ScanMultiKernelSMEM wrapper for preallocated benchmark version.

template<int BLOCK_SIZE>
void ScanMultiKernelSMEMRecursive(int* d_input, int* d_output, int n)
{
    const int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Base case: single tile
    if (num_tiles == 1) {
        ScanTilesSMEM<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_input, d_output, n, nullptr);
        CHECK_CUDA(cudaGetLastError());
        return;
    }

    // Allocate tile aggregates
    int* d_tile_aggregates;
    int* d_tile_aggregates_scanned;
    CHECK_CUDA(cudaMalloc(&d_tile_aggregates, num_tiles * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tile_aggregates_scanned, num_tiles * sizeof(int)));

    // Phase 1: Scan tiles, extract aggregates
    ScanTilesSMEM<BLOCK_SIZE><<<num_tiles, BLOCK_SIZE>>>(
        d_input, d_output, n, d_tile_aggregates);
    CHECK_CUDA(cudaGetLastError());

    // Phase 2: Recursively scan aggregates
    ScanMultiKernelSMEMRecursive<BLOCK_SIZE>(d_tile_aggregates, d_tile_aggregates_scanned, num_tiles);
    
    // Phase 3: Add prefixes
    AddTilePrefixes<<<num_tiles, BLOCK_SIZE>>>(d_output, n, d_tile_aggregates_scanned);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(d_tile_aggregates));
    CHECK_CUDA(cudaFree(d_tile_aggregates_scanned));
}

// ============================================================================
// MULTI-KERNEL SCAN: WARP SHUFFLE (RECURSIVE - EDUCATIONAL)
// ============================================================================
// Same algorithm as above but uses warp shuffle for tile scans.

template<int BLOCK_SIZE>
void ScanMultiKernelWarpRecursive(int* d_input, int* d_output, int n)
{
    const int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Base case: single tile
    if (num_tiles == 1) {
        ScanTilesWarp<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_input, d_output, n, nullptr);
        CHECK_CUDA(cudaGetLastError());
        return;
    }

    // Allocate tile aggregates
    int* d_tile_aggregates;
    int* d_tile_aggregates_scanned;
    CHECK_CUDA(cudaMalloc(&d_tile_aggregates, num_tiles * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tile_aggregates_scanned, num_tiles * sizeof(int)));

    // Phase 1: Scan tiles, extract aggregates
    ScanTilesWarp<BLOCK_SIZE><<<num_tiles, BLOCK_SIZE>>>(
        d_input, d_output, n, d_tile_aggregates);
    CHECK_CUDA(cudaGetLastError());

    // Phase 2: Recursively scan aggregates
    ScanMultiKernelWarpRecursive<BLOCK_SIZE>(d_tile_aggregates, d_tile_aggregates_scanned, num_tiles);
    
    // Phase 3: Add prefixes
    AddTilePrefixes<<<num_tiles, BLOCK_SIZE>>>(d_output, n, d_tile_aggregates_scanned);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(d_tile_aggregates));
    CHECK_CUDA(cudaFree(d_tile_aggregates_scanned));
}

// ============================================================================
// MULTI-KERNEL SCAN: PREALLOCATED VERSIONS (FOR BENCHMARKS)
// ============================================================================
// Same algorithm but uses preallocated temp buffer.
// Offset tracks position in temp buffer across recursive calls.

template<int BLOCK_SIZE>
void ScanMultiKernelSMEMPrealloc(int* d_input, int* d_output, int n,
                                  int* d_temp, size_t& offset)
{
    const int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Base case: single tile
    if (num_tiles == 1) {
        ScanTilesSMEM<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_input, d_output, n, nullptr);
        CHECK_CUDA(cudaGetLastError());
        return;
    }

    // Carve out space from temp buffer
    int* d_tile_aggregates = d_temp + offset;
    offset += num_tiles;
    int* d_tile_aggregates_scanned = d_temp + offset;
    offset += num_tiles;

    // Phase 1: Scan tiles, extract aggregates
    ScanTilesSMEM<BLOCK_SIZE><<<num_tiles, BLOCK_SIZE>>>(
        d_input, d_output, n, d_tile_aggregates);
    CHECK_CUDA(cudaGetLastError());

    // Phase 2: Recursively scan aggregates
    ScanMultiKernelSMEMPrealloc<BLOCK_SIZE>(d_tile_aggregates, d_tile_aggregates_scanned, 
                                             num_tiles, d_temp, offset);
    
    // Phase 3: Add prefixes
    AddTilePrefixes<<<num_tiles, BLOCK_SIZE>>>(d_output, n, d_tile_aggregates_scanned);
    CHECK_CUDA(cudaGetLastError());
}

template<int BLOCK_SIZE>
void ScanMultiKernelWarpPrealloc(int* d_input, int* d_output, int n,
                                  int* d_temp, size_t& offset)
{
    const int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Base case: single tile
    if (num_tiles == 1) {
        ScanTilesWarp<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_input, d_output, n, nullptr);
        CHECK_CUDA(cudaGetLastError());
        return;
    }

    // Carve out space from temp buffer
    int* d_tile_aggregates = d_temp + offset;
    offset += num_tiles;
    int* d_tile_aggregates_scanned = d_temp + offset;
    offset += num_tiles;

    // Phase 1: Scan tiles, extract aggregates
    ScanTilesWarp<BLOCK_SIZE><<<num_tiles, BLOCK_SIZE>>>(
        d_input, d_output, n, d_tile_aggregates);
    CHECK_CUDA(cudaGetLastError());

    // Phase 2: Recursively scan aggregates
    ScanMultiKernelWarpPrealloc<BLOCK_SIZE>(d_tile_aggregates, d_tile_aggregates_scanned, 
                                             num_tiles, d_temp, offset);
    
    // Phase 3: Add prefixes
    AddTilePrefixes<<<num_tiles, BLOCK_SIZE>>>(d_output, n, d_tile_aggregates_scanned);
    CHECK_CUDA(cudaGetLastError());
}

// ============================================================================
// BENCHMARK WRAPPERS
// ============================================================================
// Wrappers for RunBenchmark<> that handle temp storage calculation.

template<int BLOCK_SIZE>
struct ScanMultiKernelSMEM {
    static size_t GetTempSize(int n) {
        size_t total = 0;
        int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        while (num_tiles > 1) {
            total += 2 * num_tiles * sizeof(int);
            num_tiles = (num_tiles + BLOCK_SIZE - 1) / BLOCK_SIZE;
        }
        return total;
    }

    static void Run(int* d_input, int* d_output, int n, void* d_temp) {
        size_t offset = 0;
        ScanMultiKernelSMEMPrealloc<BLOCK_SIZE>(d_input, d_output, n, 
                                                 static_cast<int*>(d_temp), offset);
    }
};

template<int BLOCK_SIZE>
struct ScanMultiKernelWarp {
    static size_t GetTempSize(int n) {
        size_t total = 0;
        int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        while (num_tiles > 1) {
            total += 2 * num_tiles * sizeof(int);
            num_tiles = (num_tiles + BLOCK_SIZE - 1) / BLOCK_SIZE;
        }
        return total;
    }

    static void Run(int* d_input, int* d_output, int n, void* d_temp) {
        size_t offset = 0;
        ScanMultiKernelWarpPrealloc<BLOCK_SIZE>(d_input, d_output, n, 
                                                 static_cast<int*>(d_temp), offset);
    }
};