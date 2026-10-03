#include "compaction.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define SCAN_SHARED 1  // 1 = shared memory multi block scan, 0 = global memory scan
#define EFFICIENT_BLOCK_SIZE 512
#define SMEM_BLOCK_SIZE 512
#define SMEM_ELEMS (2 * SMEM_BLOCK_SIZE)  // each thread handles 2 elements

#define SCAN_AVOID_BANK_CONFLICTS 1
#define LOG_NUM_BANKS 5  // 32 banks
#if SCAN_AVOID_BANK_CONFLICTS
// one empty slot every 32 elements, so indices 32 apart land in neighboring banks instead of the same one
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#else
#define CONFLICT_FREE_OFFSET(n) 0
#endif

namespace Compaction
{
    static int* dev_bools = NULL;
    static int* dev_indices = NULL;
    static PathSegment* dev_scratch = NULL;
    static std::vector<int*> dev_blockSums;  // one per recursion level

    // inline helpers
    inline int ilog2(int x) {
        int lg = 0;
        while (x >>= 1) {
            ++lg;
        }
        return lg;
    }

    inline int ilog2ceil(int x) {
        return x == 1 ? 0 : ilog2(x - 1) + 1;
    }

    /// MAIN FUNCTIONS (no shared mem)

    // only launch one thread per active node, so idx gets stretched back out to the array index k
    __global__ void kernUpSweep(int active, int stride, int *data) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= active) return;

        int k = idx * stride * 2;
        data[k + stride * 2 - 1] += data[k + stride - 1];  // absorb element from last iteration
    }

    __global__ void kernDownSweep(int active, int stride, int *data) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= active) return;

        int k = idx * stride * 2;
        int t = data[k + stride - 1];
        data[k + stride - 1] = data[k + stride * 2 - 1];  // swap
        data[k + stride * 2 - 1] += t;  // absorb
    }

    // in place exclusive scan = upsweep then downsweep, on an array already padded to a power of 2
    void scanDevice(int n_padded, int *dev_idata) {
        int levels = ilog2(n_padded);

        for (int d = 0; d < levels; d++) {
            int active = n_padded >> (d + 1);
            dim3 fullBlocksPerGrid((active + EFFICIENT_BLOCK_SIZE - 1) / EFFICIENT_BLOCK_SIZE);
            kernUpSweep<<<fullBlocksPerGrid, EFFICIENT_BLOCK_SIZE>>>(active, 1 << d, dev_idata);
        }

        // set the last element to 0 before downsweep
        cudaMemset(dev_idata + n_padded - 1, 0, sizeof(int));

        for (int d = levels - 1; d >= 0; d--) {
            int active = n_padded >> (d + 1);
            dim3 fullBlocksPerGrid((active + EFFICIENT_BLOCK_SIZE - 1) / EFFICIENT_BLOCK_SIZE);
            kernDownSweep<<<fullBlocksPerGrid, EFFICIENT_BLOCK_SIZE>>>(active, 1 << d, dev_idata);
        }
    }

    /// SHARED MEMORY SCAN (gpu gems 3 ch 39)

    // exclusive scan of one SMEM_ELEMS chunk per block. each block's total goes to blockSums
    // only one kernel launch, not 2*levels! syncthreads enforces synchronization vs kernel completion like the gmem version
    __global__ void kernBlockScan(int n, int* data, int* blockSums)
    {
        __shared__ int temp[SMEM_ELEMS + CONFLICT_FREE_OFFSET(SMEM_ELEMS)];  // add in the padding to avoid bank conflicts.
        int thid = threadIdx.x;
        int base = blockIdx.x * SMEM_ELEMS;

        // each thread handles 2 elements because the scan is a binary tree
        // only exist to index into mem. thid and thid + SMEM_BLOCK_SIZE so the global loads will coalesce. past the end loads 0, so only the last block gets padded instead of the whole array
        int ai = thid;
        int bi = thid + SMEM_BLOCK_SIZE;
        int bankOffsetA = CONFLICT_FREE_OFFSET(ai);  // make sure to increment the bank offset
        int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
        temp[ai + bankOffsetA] = base + ai < n ? data[base + ai] : 0;
        temp[bi + bankOffsetB] = base + bi < n ? data[base + bi] : 0;

        // same indexing as kernUpSweep. thid plays the role of idx
        int stride = 1;
        for (int active = SMEM_ELEMS >> 1; active > 0; active >>= 1)  // same logic as before. half the active...
        {
            __syncthreads();
            if (thid < active)
            {
                int k = thid * stride * 2;
                int a = k + stride - 1;
                int b = k + stride * 2 - 1;
                a += CONFLICT_FREE_OFFSET(a);  // need to add every sweep iteration
                b += CONFLICT_FREE_OFFSET(b);
                temp[b] += temp[a];
            }
            stride <<= 1;  // ...double the stride
        }

        // save + set the last element to 0 before downsweep
        if (thid == 0)
        {
            int last = SMEM_ELEMS - 1 + CONFLICT_FREE_OFFSET(SMEM_ELEMS - 1);
            blockSums[blockIdx.x] = temp[last];
            temp[last] = 0;
        }

        // downsweep
        for (int active = 1; active < SMEM_ELEMS; active <<= 1)
        {
            stride >>= 1;
            __syncthreads();
            if (thid < active)
            {
                int k = thid * stride * 2;
                int a = k + stride - 1;
                int b = k + stride * 2 - 1;
                a += CONFLICT_FREE_OFFSET(a);  // need to add every sweep iteration
                b += CONFLICT_FREE_OFFSET(b);
                int t = temp[a];
                temp[a] = temp[b];
                temp[b] += t;
            }
        }
        __syncthreads();

        // write out. also avoids conflicts
        if (base + ai < n) data[base + ai] = temp[ai + bankOffsetA];
        if (base + bi < n) data[base + bi] = temp[bi + bankOffsetB];
    }

    // each chunk was scanned from 0, so add the scanned block sums to shift them to the right start
    __global__ void kernAddBlockOffsets(int n, int* data, const int* blockOffsets)
    {
        int base = blockIdx.x * SMEM_ELEMS;
        int offset = blockOffsets[blockIdx.x];
        int ai = base + threadIdx.x;
        int bi = base + threadIdx.x + SMEM_BLOCK_SIZE;
        if (ai < n) data[ai] += offset;
        if (bi < n) data[bi] += offset;
    }

    // in place exclusive scan of data[0, n). no padding needed
    void scanShared(int n, int* data, int level)
    {
        int blocks = (n + SMEM_ELEMS - 1) / SMEM_ELEMS;
        kernBlockScan<<<blocks, SMEM_BLOCK_SIZE>>>(n, data, dev_blockSums[level]);

        // here is the recursive scan across the blocksums. 
        if (blocks > 1)
        {
            scanShared(blocks, dev_blockSums[level], level + 1);
            kernAddBlockOffsets<<<blocks, SMEM_BLOCK_SIZE>>>(n, data, dev_blockSums[level]);
        }
    }

    /// PATH SEGMENT PARTITION

    __global__ void kernMapAlive(int n, const PathSegment* paths, int* bools)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx < n)
        {
            bools[idx] = paths[idx].remainingBounces > 0;
        }
    }

    // kernScatter but dead paths get kept too. idx - indices[idx] = how many dead paths are before idx
    __global__ void kernScatterPartition(int n, int numAlive, const PathSegment* in, PathSegment* out,
                                         const int* bools, const int* indices)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx < n)
        {
            int dst = bools[idx] ? indices[idx] : numAlive + (idx - indices[idx]);
            out[dst] = in[idx];
        }
    }

    void init(int maxN)
    {
        // setup
        int n_padded = 1 << ilog2ceil(maxN);
        cudaMalloc(&dev_bools, maxN * sizeof(int));
        cudaMalloc(&dev_indices, n_padded * sizeof(int));
        cudaMalloc(&dev_scratch, maxN * sizeof(PathSegment));

        int n = maxN;
        while (true)
        {
            int blocks = (n + SMEM_ELEMS - 1) / SMEM_ELEMS;
            int* sums;
            cudaMalloc(&sums, blocks * sizeof(int));
            dev_blockSums.push_back(sums);
            if (blocks <= 1) break;
            n = blocks;
        }
    }

    void free()
    {
        cudaFree(dev_bools);
        cudaFree(dev_indices);
        cudaFree(dev_scratch);
        for (int* sums : dev_blockSums)
        {
            cudaFree(sums);
        }
        dev_blockSums.clear();
    }

    int partitionPaths(int n, PathSegment* paths)
    {
        if (n <= 0)
        {
            return 0;
        }
        dim3 fullBlocksPerGrid((n + EFFICIENT_BLOCK_SIZE - 1) / EFFICIENT_BLOCK_SIZE);

        kernMapAlive<<<fullBlocksPerGrid, EFFICIENT_BLOCK_SIZE>>>(n, paths, dev_bools);
        cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
#if SCAN_SHARED
        scanShared(n, dev_indices, 0);
#else
        int n_padded = 1 << ilog2ceil(n);
        cudaMemset(dev_indices + n, 0, (n_padded - n) * sizeof(int));
        scanDevice(n_padded, dev_indices);
#endif

        int lastBool, lastIndex;
        cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
        int numAlive = lastBool + lastIndex;

        kernScatterPartition<<<fullBlocksPerGrid, EFFICIENT_BLOCK_SIZE>>>(n, numAlive, paths, dev_scratch, dev_bools, dev_indices);

        // copy back instead of swapping buffers
        cudaMemcpy(paths, dev_scratch, n * sizeof(PathSegment), cudaMemcpyDeviceToDevice);

        return numAlive;
    }
}
