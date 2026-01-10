template <int BLOCK_SIZE>
__global__ void ScanDecoupledLookbackWarpKernelClaude(
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

    int tile_idx = s_tile_idx;
    int gid = tile_idx * BLOCK_SIZE + threadIdx.x;

    // Step 2: Load and scan block
    int value = (gid < n) ? input[gid] : 0;
    value = BlockScan<BLOCK_SIZE>(value);

    // Step 3: Last warp does the decoupled lookback
    int warp_idx = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    constexpr int last_warp_idx = BLOCK_SIZE / warpSize - 1;

    if (warp_idx == last_warp_idx) {
        // Get block aggregate from last thread's value
        int block_aggregate = __shfl_sync(0xFFFFFFFF, value, warpSize - 1);

        // Publish aggregate (only one thread writes)
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
            int lookback_base = tile_idx - 1;  // Closest predecessor

            while (true) {
                int my_lookback_idx = lookback_base - lane;

                // Each lane spins on its own predecessor
                TileDescriptor pred_info;
                pred_info.value = 0;
                pred_info.status = TileStatus::PREFIX;  // Default for out-of-bounds

                if (my_lookback_idx >= 0) {
                    do {
                        pred_info.raw = atomicAdd(&tile_descriptors[my_lookback_idx].raw, 0);
                    } while (pred_info.status == TileStatus::INVALID);
                }

                // Which lanes found PREFIX?
                unsigned prefix_mask = __ballot_sync(0xFFFFFFFF, pred_info.status == TileStatus::PREFIX);

                // Find closest PREFIX (lowest lane = highest lookback_idx)
                int prefix_lane = __ffs(prefix_mask) - 1;

                // Sum values from lane 0 through prefix_lane using inclusive scan
                int my_contribution = (lane <= prefix_lane) ? pred_info.value : 0;

                // Warp inclusive scan
                for (int offset = 1; offset < warpSize; offset *= 2) {
                    int other = __shfl_up_sync(0xFFFFFFFF, my_contribution, offset);
                    if (lane >= offset) {
                        my_contribution += other;
                    }
                }

                // prefix_lane has sum of lanes 0..prefix_lane
                int iteration_sum = __shfl_sync(0xFFFFFFFF, my_contribution, prefix_lane);
                exclusive_prefix += iteration_sum;

                // If we hit a PREFIX, we're done
                if (prefix_lane < warpSize - 1 || (my_lookback_idx <= 0 && lane == prefix_lane)) {
                    // Either found PREFIX mid-warp, or reached tile 0
                    break;
                }

                // All 32 were AGGREGATE—shift back and continue
                lookback_base -= warpSize;
            }

            if (lane == 0) {
                s_prefix = exclusive_prefix;
            }

            // Upgrade to PREFIX
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

    // Step 4: All threads add prefix and write
    if (gid < n) {
        output[gid] = s_prefix + value;
    }
}