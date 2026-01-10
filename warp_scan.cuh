// warp_scan.cuh

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