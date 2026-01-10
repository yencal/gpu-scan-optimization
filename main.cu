// main.cu
// Test infrastructure for comparing scan algorithms

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <utils.cuh>

#define CHECK_CUDA(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

void Initialization(int *buffer, int *gold, int n)
{
    for (int i = 0; i < n; ++i)
    {
        buffer[i] = 1;
    }

    gold[0] = buffer[0];
    for (int i = 1; i < n; ++i)
    {
        gold[i] = gold[i - 1] + buffer[i];
    }
}

void VerifyScan(const int* output, const int* gold, int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (output[i] != gold[i])
        {
            std::cerr << "Mismatch at index " << i << ": "
                      << "got " << output[i] << ", expected " << gold[i] << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

float GetPeakBandwidth()
{
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    
    int memory_clock_khz;
    int memory_bus_width_bits;
    
    CHECK_CUDA(cudaDeviceGetAttribute(&memory_clock_khz, 
                                      cudaDevAttrMemoryClockRate, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&memory_bus_width_bits, 
                                      cudaDevAttrGlobalMemoryBusWidth, device));
    
    float peak_bandwidth_gbs = 2.0f * memory_clock_khz * 
                               (memory_bus_width_bits / 8.0f) / 1e6f;
    
    return peak_bandwidth_gbs;
}

// ============================================================================
// WARP AND BLOCK SCAN PRIMITIVES
// ============================================================================

static __device__ __forceinline__ int WarpScan(int value)
{
    int lane = threadIdx.x % warpSize;
    for (int offset = 1; offset < warpSize; offset *= 2) {
        int tmp = __shfl_up_sync(0xFFFFFFFF, value, offset);
        if (lane >= offset) {
            value += tmp;
        }
    }
    return value;
}

template<int BLOCK_SIZE>
static __device__ __forceinline__ int BlockScan(int value)
{
    int wid = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    constexpr int num_warps = (BLOCK_SIZE + 31) / 32;

    int warp_scan = WarpScan(value);

    __shared__ int smem[num_warps];
    if (lane == warpSize-1) {
        smem[wid] = warp_scan;
    }
    __syncthreads();

    if (wid == 0) {
        int tmp = (threadIdx.x < num_warps) ? smem[threadIdx.x] : 0;
        tmp = WarpScan(tmp);
        if (threadIdx.x < num_warps) {
            smem[threadIdx.x] = tmp;
        }
    }
    __syncthreads();

    int warp_prefix = (wid > 0) ? smem[wid - 1] : 0;
    int block_scan = warp_scan + warp_prefix;
    return block_scan;
}

template<int BLOCK_SIZE>
static __device__ __forceinline__ int BlockScanExclusive(int value)
{
    int inclusive = BlockScan<BLOCK_SIZE>(value);
    return inclusive - value;
}

// ============================================================================
// KERNELS
// ============================================================================

template<int BLOCK_SIZE>
__global__ void ScanBlocksWarp(
    const int* __restrict__ input,
    int* __restrict__ output,
    int n,
    int* block_sums)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int value = (gid < n) ? input[gid] : 0;
    value = BlockScan<BLOCK_SIZE>(value);

    if (gid < n) {
        output[gid] = value;
    }

    if (threadIdx.x == BLOCK_SIZE - 1 && block_sums != nullptr) {
        block_sums[blockIdx.x] = value;
    }
}



template<int BLOCK_SIZE>
__global__ void ScanBlocksSMEM(
    const int* __restrict__ input,
    int* __restrict__ output,
    int n,
    int* block_sums)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ int smem[2][BLOCK_SIZE];

    int wbuf = 0;
    int rbuf = 1;
    smem[wbuf][tid] = (gid < n) ? input[gid] : int(0);
    __syncthreads();

    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        wbuf = 1 - wbuf;
        rbuf = 1 - wbuf;
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
    
    if (tid == BLOCK_SIZE - 1 && block_sums != nullptr) {
        block_sums[blockIdx.x] = smem[wbuf][BLOCK_SIZE - 1];
    }
}

__global__ void AddBlockSums(int* output, int n, const int* scanned_block_sums)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0 && gid < n) {
        output[gid] += scanned_block_sums[blockIdx.x - 1];
    }
}

// ============================================================================
// MULTI-KERNEL SCAN ALGORITHMS
// ============================================================================

template<int BLOCK_SIZE>
void ScanMultiKernelBaseline(int* d_input, int* d_output, int n)
{
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (num_blocks == 1) {
        ScanBlocksSMEM<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_input, d_output, n, nullptr);
        CHECK_CUDA(cudaGetLastError());
        return;
    }

    int* d_block_sums;
    int* d_block_sums_scanned;
    CHECK_CUDA(cudaMalloc(&d_block_sums, num_blocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_block_sums_scanned, num_blocks * sizeof(int)));

    ScanBlocksSMEM<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, n, d_block_sums);
    CHECK_CUDA(cudaGetLastError());

    ScanMultiKernelBaseline<BLOCK_SIZE>(d_block_sums, d_block_sums_scanned, num_blocks);
    
    AddBlockSums<<<num_blocks, BLOCK_SIZE>>>(d_output, n, d_block_sums_scanned);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(d_block_sums));
    CHECK_CUDA(cudaFree(d_block_sums_scanned));
}

template<int BLOCK_SIZE>
void ScanMultiKernelWarp(int* d_input, int* d_output, int n)
{
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (num_blocks == 1) {
        ScanBlocksWarp<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d_input, d_output, n, nullptr);
        CHECK_CUDA(cudaGetLastError());
        return;
    }

    int* d_block_sums;
    int* d_block_sums_scanned;
    CHECK_CUDA(cudaMalloc(&d_block_sums, num_blocks * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_block_sums_scanned, num_blocks * sizeof(int)));

    ScanBlocksWarp<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, n, d_block_sums);
    CHECK_CUDA(cudaGetLastError());

    ScanMultiKernelWarp<BLOCK_SIZE>(d_block_sums, d_block_sums_scanned, num_blocks);
    
    AddBlockSums<<<num_blocks, BLOCK_SIZE>>>(d_output, n, d_block_sums_scanned);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(d_block_sums));
    CHECK_CUDA(cudaFree(d_block_sums_scanned));
}

// ============================================================================
// SINGLE-KERNEL SCAN ALGORITHMS (TO BE IMPLEMENTED)
// ============================================================================

void ScanChained(int* d_input, int* d_output, int n)
{
    // TODO: Implement chained scan
    // - Each block waits for previous block to finish
    // - Uses atomics for synchronization
    // - Expected to be slow due to serialization
    std::cerr << "ScanChained not yet implemented" << std::endl;
    std::exit(EXIT_FAILURE);
}

void ScanLookbackBlock(int* d_input, int* d_output, int n)
{
    // TODO: Implement decoupled lookback with single-thread lookback
    // - Each block scans independently
    // - Single thread per block does lookback
    // - Uses INVALID/AGGREGATE/PREFIX states
    std::cerr << "ScanLookbackBlock not yet implemented" << std::endl;
    std::exit(EXIT_FAILURE);
}

void ScanLookbackWarp(int* d_input, int* d_output, int n)
{
    // TODO: Implement decoupled lookback with warp-level lookback
    // - Entire warp cooperates on lookback (32x parallelism)
    // - Uses __ballot_sync and warp reductions
    // - Expected to be fastest
    std::cerr << "ScanLookbackWarp not yet implemented" << std::endl;
    std::exit(EXIT_FAILURE);
}

void ScanLookbackWarpCutoff(int* d_input, int* d_output, int n)
{
    // TODO: Implement warp lookback with cutoff optimization
    // - Same as ScanLookbackWarp but with cutoff
    // - Prevents worst-case deep lookback chains
    // - Fallback to serial scan after CUTOFF blocks
    std::cerr << "ScanLookbackWarpCutoff not yet implemented" << std::endl;
    std::exit(EXIT_FAILURE);
}

template<typename ScanFunc>
void RunTest(
    const char* label,
    ScanFunc scan_func,
    int n,
    float peak_bandwidth,
    int warmup_runs = 2,
    int timed_runs = 10)
{
    std::cout << "\n================================================" << std::endl;
    std::cout << "Testing: " << label << std::endl;
    std::cout << "================================================" << std::endl;

    // Initialize host memory
    int* h_input = new int[n];
    int* h_gold = new int[n];
    Initialization(h_input, h_gold, n);

    // Allocate device memory
    int* d_input;
    int* d_output;
    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice));

    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        scan_func(d_input, d_output, n);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    // Validate first (fail fast)
    int* h_output = new int[n];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));
    VerifyScan(h_output, h_gold, n);

    // Timed runs
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::vector<float> times;
    for (int i = 0; i < timed_runs; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        scan_func(d_input, d_output, n);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float time_ms;
        CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
        times.push_back(time_ms);
    }

    // Calculate statistics
    float min_time = times[0];
    float max_time = times[0];
    float sum_time = 0.0f;
    for (float t : times) {
        min_time = (t < min_time) ? t : min_time;
        max_time = (t > max_time) ? t : max_time;
        sum_time += t;
    }
    float avg_time = sum_time / timed_runs;

    // Calculate bandwidth
    size_t bytes = 2 * n * sizeof(int);
    float bandwidth_gbs = (bytes / 1e9f) / (avg_time / 1000.0f);
    float percent_peak = (bandwidth_gbs / peak_bandwidth) * 100.0f;

    // Print results
    std::cout << "Time (avg/min/max): " 
              << avg_time << " / " << min_time << " / " << max_time << " ms" << std::endl;
    std::cout << "Bandwidth: " << bandwidth_gbs << " GB/s (" 
              << percent_peak << "% of peak)" << std::endl;

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    delete[] h_input;
    delete[] h_output;
    delete[] h_gold;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv)
{
    size_t num_elements = size_t{1} << 30;

    if (argc >= 2) {
        int power = std::atoi(argv[1]);
        if (power < 1 || power > 30) {
            std::cerr << "Power must be between 1 and 30" << std::endl;
            return EXIT_FAILURE;
        }
        num_elements = size_t{1} << power;
    }

    std::cout << "Array size: " << num_elements << " elements (" 
              << (num_elements * sizeof(int)) / 1e9 << " GB)" << std::endl;

    float peak_bandwidth = GetPeakBandwidth();
    std::cout << "Device peak bandwidth: " << peak_bandwidth << " GB/s" << std::endl;

    // Run all tests
    RunTest("Multi-kernel Baseline (SMEM)", ScanMultiKernelBaseline, num_elements, peak_bandwidth);
    RunTest("Multi-kernel Warp Primitives", ScanMultiKernelWarp, num_elements, peak_bandwidth);

    return EXIT_SUCCESS;
}