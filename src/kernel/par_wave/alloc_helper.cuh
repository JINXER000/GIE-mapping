
#ifndef SRC_ALLOC_HELPER_H
#define SRC_ALLOC_HELPER_H

#include "voxmap_utils.cuh"
namespace vhashing {


    /**
 * Use by AllocKeys for bulk allocations.
 *
 * */
    template <class HashTable>
    struct RequiresAllocation {
        HashTable bm;
        typedef typename HashTable::KeyType Key;
        __device__
        bool operator() (const Key &k) {
            return !bm.isequal(k, bm.EmptyKey()) && bm.find(k) == bm.end();
        }
    };

/**
 * For each key, try to allocate it in hashtable.
 *
 * Whether it fails/succeeds due to contention, record the result
 * in success.
 *
 * */
    __global__
    void TryAllocateKernel(
            HASH_BASE hash_base,
            int3 *keys,
            int *success,
            int blockBase,
            int numJobs
    ) {
        int idx_1d = blockDim.x * blockIdx.x + threadIdx.x;

        if (idx_1d >= numJobs)
            return;

        int alloc_id = hash_base.insert_to_id(keys[idx_1d], blockBase + idx_1d);
        if(alloc_id == -1)
        {
            success[idx_1d] = hash_base.alloc.offsets[blockBase + idx_1d];
        }else
        {
            success[idx_1d] = -1;
        }
    }

/**
 * If allocation failed due to contention for some keys,
 * then for each of these keys, free the allocated block,
 * and record which keys failed.
 * */
    __global__
    void ReturnAllocations(
            HASH_BASE self,
            int *success,
            int *unsuccessful,
            int numJobs) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;

        if (x >= numJobs)
            return;

        if (success[x] != -1) {
            *unsuccessful = 1;
            self.alloc.free(success[x]);
        }
    }


}


#endif //SRC_ALLOC_HELPER_H
