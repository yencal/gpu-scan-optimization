// scan_cub.cuh

#include <cub/cub.cuh>

struct ScanCUB {
    static size_t GetTempSize(int n) {
        size_t temp_size = 0;
        cub::DeviceScan::InclusiveSum(nullptr, temp_size, 
            (int*)nullptr, (int*)nullptr, n);
        return temp_size;
    }

    static void Run(int* d_input, int* d_output, int n, void* d_temp) {
        size_t temp_size = GetTempSize(n);
        cub::DeviceScan::InclusiveSum(d_temp, temp_size, 
            d_input, d_output, n);
    }
};