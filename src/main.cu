// main.cu
// Benchmark runner for scan algorithm comparison

#include <iostream>
#include <cstdlib>

#include "utils.cuh"
#include "scan_multi_kernel.cuh"
#include "scan_chained.cuh"
#include "scan_lookback.cuh"

int main(int argc, char** argv)
{
    constexpr int BLOCK_SIZE = 256;

    // Default: 2^28 elements (~1GB for int32)
    int power = 28;
    
    if (argc >= 2) {
        power = std::atoi(argv[1]);
        if (power < 1 || power > 30) {
            std::cerr << "Power must be between 1 and 30" << std::endl;
            return EXIT_FAILURE;
        }
    }

    const int n = 1 << power;
    
    std::cout << "========================================" << std::endl;
    std::cout << "GPU Scan Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Array size: 2^" << power << " = " << n << " elements" << std::endl;
    std::cout << "Data size: " << (static_cast<size_t>(n) * sizeof(int)) / (1024.0 * 1024.0 * 1024.0) 
              << " GB" << std::endl;
    std::cout << "Block size: " << BLOCK_SIZE << std::endl;

    const float peak_bandwidth = GetPeakBandwidth();
    std::cout << "Device peak bandwidth: " << peak_bandwidth << " GB/s" << std::endl;

    // ========================================================================
    // Multi-kernel approaches
    // ========================================================================

    RunBenchmark<ScanMultiKernelSMEM<BLOCK_SIZE>>(
        "Multi-kernel (SMEM baseline)", n, peak_bandwidth);

    RunBenchmark<ScanMultiKernelWarp<BLOCK_SIZE>>(
        "Multi-kernel (Warp shuffle)", n, peak_bandwidth);

    // ========================================================================
    // Single-pass approaches
    // ========================================================================

    RunBenchmark<ScanChained<BLOCK_SIZE>>(
        "Chained scan (serialized)", n, peak_bandwidth);

    RunBenchmark<ScanLookbackSingleThread<BLOCK_SIZE>>(
        "Lookback (single-thread)", n, peak_bandwidth);

    RunBenchmark<ScanLookbackWarp<BLOCK_SIZE>>(
        "Lookback (warp)", n, peak_bandwidth);

    // ========================================================================
    // Coarsening sweep
    // ========================================================================

    RunBenchmark<ScanLookbackWarpCoarsened<BLOCK_SIZE, 2>>(
        "Lookback (warp + coarsened x2)", n, peak_bandwidth);

    RunBenchmark<ScanLookbackWarpCoarsened<BLOCK_SIZE, 4>>(
        "Lookback (warp + coarsened x4)", n, peak_bandwidth);

    RunBenchmark<ScanLookbackWarpCoarsened<BLOCK_SIZE, 8>>(
        "Lookback (warp + coarsened x8)", n, peak_bandwidth);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return EXIT_SUCCESS;
}
