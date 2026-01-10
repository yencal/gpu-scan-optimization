// utils.cuh
// Error checking and benchmark utilities

#pragma once

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================

#define CHECK_CUDA(val) CheckCuda((val), #val, __FILE__, __LINE__)

inline void CheckCuda(cudaError_t err, const char* const func, 
                      const char* const file, const int line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// ============================================================================
// DEVICE INFO
// ============================================================================

inline float GetPeakBandwidth()
{
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    
    int memory_clock_khz;
    int memory_bus_width_bits;
    
    CHECK_CUDA(cudaDeviceGetAttribute(&memory_clock_khz, 
                                      cudaDevAttrMemoryClockRate, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&memory_bus_width_bits, 
                                      cudaDevAttrGlobalMemoryBusWidth, device));
    
    // DDR: multiply by 2
    float peak_bandwidth_gbs = 2.0f * memory_clock_khz * 
                               (memory_bus_width_bits / 8.0f) / 1e6f;
    
    return peak_bandwidth_gbs;
}

// ============================================================================
// TEST DATA INITIALIZATION AND VERIFICATION
// ============================================================================

inline void InitializeTestData(int* input, int* gold, int n)
{
    for (int i = 0; i < n; ++i) {
        input[i] = 1;
    }

    gold[0] = input[0];
    for (int i = 1; i < n; ++i) {
        gold[i] = gold[i - 1] + input[i];
    }
}

inline bool VerifyScan(const int* output, const int* gold, int n)
{
    for (int i = 0; i < n; ++i) {
        if (output[i] != gold[i]) {
            std::cerr << "Mismatch at index " << i << ": "
                      << "got " << output[i] << ", expected " << gold[i] << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// BENCHMARK RUNNER
// ============================================================================
// ScanAlgorithm must provide:
//   static size_t GetTempSize(int n)
//   static void Run(int* d_input, int* d_output, int n, void* temp_storage)

template<typename ScanAlgorithm>
void RunBenchmark(
    const char* label,
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
    InitializeTestData(h_input, h_gold, n);

    // Allocate device memory
    int* d_input;
    int* d_output;
    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate temp storage ONCE outside timing loop
    size_t temp_size = ScanAlgorithm::GetTempSize(n);
    void* d_temp = nullptr;
    if (temp_size > 0) {
        CHECK_CUDA(cudaMalloc(&d_temp, temp_size));
    }

    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        ScanAlgorithm::Run(d_input, d_output, n, d_temp);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Verify correctness (fail fast)
    int* h_output = new int[n];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));
    if (!VerifyScan(h_output, h_gold, n)) {
        std::cerr << "FAILED: " << label << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "Correctness: PASSED" << std::endl;

    // Timed runs
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::vector<float> times;
    for (int i = 0; i < timed_runs; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        ScanAlgorithm::Run(d_input, d_output, n, d_temp);
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

    // Calculate bandwidth (read input + write output)
    size_t bytes = 2 * static_cast<size_t>(n) * sizeof(int);
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
    if (d_temp) CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    delete[] h_input;
    delete[] h_output;
    delete[] h_gold;
}