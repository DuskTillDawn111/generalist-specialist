// Host-only wrapper. No CUDA runtime/driver library is linked or called.
#include <cuda_occupancy.h>
#include <iostream>
#include <string>

int main() {
    cudaOccDeviceProp p;
    cudaOccFuncAttributes f;
    cudaOccDeviceState s;
    cudaOccResult r = {};
    int gc, shmem, cache, block;
    size_t dynamic;
    if (!(std::cin >> p.computeMajor >> p.computeMinor >> p.maxThreadsPerBlock
          >> p.maxThreadsPerMultiprocessor >> p.regsPerBlock >> p.regsPerMultiprocessor
          >> p.warpSize >> p.sharedMemPerBlock >> p.sharedMemPerMultiprocessor
          >> p.numSms >> p.sharedMemPerBlockOptin >> p.reservedSharedMemPerBlock
          >> f.maxThreadsPerBlock >> f.numRegs >> f.sharedSizeBytes >> gc >> shmem
          >> f.maxDynamicSharedSizeBytes >> f.numBlockBarriers >> cache
          >> s.carveoutConfig >> block >> dynamic)) return 2;
    std::string extra;
    if (std::cin >> extra) return 2;
    f.partitionedGCConfig = static_cast<cudaOccPartitionedGCConfig>(gc);
    f.shmemLimitConfig = static_cast<cudaOccFuncShmemConfig>(shmem);
    s.cacheConfig = static_cast<cudaOccCacheConfig>(cache);
    const cudaOccError rc = cudaOccMaxActiveBlocksPerMultiprocessor(&r, &p, &f, &s, block, dynamic);
    std::cout << "{\"returncode\":" << static_cast<int>(rc)
              << ",\"active_blocks_per_sm\":" << r.activeBlocksPerMultiprocessor
              << ",\"allocated_registers_per_block\":" << r.allocatedRegistersPerBlock
              << ",\"allocated_shared_bytes\":" << r.allocatedSharedMemPerBlock
              << ",\"limiting_factors_mask\":" << r.limitingFactors
              << ",\"block_limits\":{\"registers\":" << r.blockLimitRegs
              << ",\"shared\":" << r.blockLimitSharedMem
              << ",\"warps\":" << r.blockLimitWarps
              << ",\"blocks\":" << r.blockLimitBlocks
              << ",\"barriers\":" << r.blockLimitBarriers
              << "},\"partitioned_gc_effective\":" << static_cast<int>(r.partitionedGCConfig)
              << "}\n";
    return 0;
}
