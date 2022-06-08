
#ifndef SRC_WAVE_CORE_CUH
#define SRC_WAVE_CORE_CUH

#include "par_wave/frontier_wrapper.h"
#include "par_wave/voxmap_utils.cuh"

// A group of local queues of node IDs, used by an entire thread block.
// Multiple queues are used to reduce memory contention.
// Thread i uses queue number (i % NUM_BIN).
template <class Ktype>
struct LocalQueues {
    // tail[n] is the index of the first empty array in elems[n]
    int tail[NUM_BIN];

    // Queue elements.
    // The contents of queue n are elems[n][0 .. tail[n] - 1].
    Ktype elems[NUM_BIN][W_QUEUE_SIZE];

    // The number of threads sharing queue n.  We use this number to
    // compute a reduction over the queue.
    int sharers[NUM_BIN];

    // Initialize or reset the queue at index 'index'.
    // Normally run in parallel for all indices.
    __device__ void reset(int index, dim3 block_dim) {
        tail[index] = 0;		// Queue contains nothing

        // Number of sharers is (threads per block / number of queues)
        // If division is not exact, assign the leftover threads to the first
        // few queues.
        sharers[index] =
                (block_dim.x >> EXP) +   // block_dim/8
                (threadIdx.x < (block_dim.x & MOD_OP));
    }
    // Append 'value' to queue number 'index'.  If queue is full, the
    // append operation fails and *overflow is set to 1.
    __device__ void append(int index, int *overflow, Ktype value) {
        // Queue may be accessed concurrently, so
        // use an atomic operation to reserve a queue index.
        int tail_index = atomicAdd(&tail[index], 1);
        if (tail_index >= W_QUEUE_SIZE)
            *overflow = 1;
        else
            elems[index][tail_index] = value;
    }

    // Perform a scan on the number of elements in queues in a a LocalQueue.
    // This function should be executed by one thread in a thread block.
    //
    // The result of the scan is used to concatenate all queues; see
    // 'concatenate'.
    //
    // The array prefix_q will hold the scan result on output:
    // [0, tail[0], tail[0] + tail[1], ...]
    //
    // The total number of elements is returned.
    __device__ int size_prefix_sum(int (&prefix_q)[NUM_BIN]) {
        prefix_q[0] = 0;
        for(int i = 1; i < NUM_BIN; i++){
            prefix_q[i] = prefix_q[i-1] + tail[i-1];
        }
        return prefix_q[NUM_BIN-1] + tail[NUM_BIN-1];
    }

    // Concatenate and copy all queues to the destination.
    // This function should be executed by all threads in a thread block.
    //
    // prefix_q should contain the result of 'size_prefix_sum'.
    __device__ void concatenate(Ktype *dst, int (&prefix_q)[NUM_BIN]) {
        // Thread n processes elems[n % NUM_BIN][n / NUM_BIN, ...]
        int q_i = threadIdx.x & MOD_OP; // w-queue index, idx of row
        int local_shift = threadIdx.x >> EXP; // shift within a w-queue, idx of col

        while(local_shift < tail[q_i]){
            dst[prefix_q[q_i] + local_shift] = elems[q_i][local_shift];

            //multiple threads are copying elements at the same time,
            //so we shift by multiple elements for next iteration
            local_shift += sharers[q_i]; // 8*64>512, so it is out of bound???
        }
    }
};


__device__  __forceinline__
void raise_outside(int3 cur_glb, int index, LocalQueues<int3> &local_q, int *overflow,
                   LocMap &loc_map, HASH_BASE &hashBase, int3* q_aux, int* q_aux_num,
                   int num_dirs, const int3* dirs_D, int gray_shade, int map_ct,
                   bool display_glb_edt, int3* changed_keys, int* changed_cnt)
{
    CrdEqualTo eq;
    if(invalid_src(cur_glb))
        return;

    int3 vb_key = get_VB_key(cur_glb);
    int alloc_id = hashBase.get_alloc_blk_id(vb_key);
    if(alloc_id== -1)
    {
        printf("voxel block not found\n");
        assert(false);
    }
    VoxelBlock* Vb= &(hashBase.alloc[alloc_id]);
    GlbVoxel* cur_vox = retrive_vox_D(cur_glb, Vb);

    // streaming
    if(display_glb_edt)
    {
        int  idx_1d = atomicAdd(changed_cnt, 1);
        assert(idx_1d<loc_map._map_volume);
        changed_keys[idx_1d] = vb_key;
    }

    bool cur_in_q = false;
    int3 valid_local_coc = cur_vox->coc_glb;
    for(int i=0; i<num_dirs; i++)
    {
        int3 nbr_glb = cur_glb + dirs_D[i];
        if(loc_map.is_inside_update_volume(loc_map.glb2loc(nbr_glb)))
            continue;

        int3 nbr_vb_key = get_VB_key(nbr_glb);
        VoxelBlock* nbr_Vb;
        // judge if nbr and cur are in the same block
        if(eq(nbr_vb_key,vb_key))
        {
            nbr_Vb = Vb;
        }else
        {
            int nbr_alloc_id = hashBase.get_alloc_blk_id(nbr_vb_key);
            if(nbr_alloc_id== -1)
                continue;
            nbr_Vb= &(hashBase.alloc[nbr_alloc_id]);
        }
        GlbVoxel* nbr_vox = retrive_vox_D(nbr_glb, nbr_Vb);
        // check if nbr is updated by UpdateHashTB(). If previous drone sees nothing, still no need to raise nbr().
        if(nbr_vox->vox_type==VOXTYPE_UNKNOWN || invalid_coc_glb(nbr_vox->coc_glb)|| invalid_dist_glb(nbr_vox->dist_sq))
            continue;
        // rule out the case that nbr_vox is also raising, or it has been raised
        if(nbr_vox->wave_layer == -map_ct || nbr_vox->update_ct == -map_ct) // seems redundant, but useful in lower_out
            continue;
        // if cur_coc is same as nbr_coc, no need to do bfs
        if(eq(nbr_vox->coc_glb, valid_local_coc))
            continue;

        bool is_nbr_raised = false;
        // tell if nbr_vox->coc_glb disappeared as well
        int3 nbr_coc_buf = loc_map.glb2loc(nbr_vox->coc_glb);
        if(loc_map.is_inside_update_volume(nbr_coc_buf))
        {
            int nbr_coc_new_val = loc_map.g_aux(nbr_coc_buf.x, nbr_coc_buf.y, nbr_coc_buf.z, 0);
            if(nbr_coc_new_val!=0) // disappeared
            {
                int cur_coc2nbr = get_squred_dist(valid_local_coc, nbr_glb);
                nbr_vox->dist_sq = cur_coc2nbr;
                nbr_vox->coc_glb = valid_local_coc;
                nbr_vox->wave_layer = -map_ct;
                nbr_vox->update_ct = -map_ct;

                local_q.append(index, overflow, nbr_glb);
                is_nbr_raised = true;

            }
        }
        // the case where the nbr cannot be raised: judge if cur can be lowered by nbr
        if(!is_nbr_raised)
        {
            int nbr_coc2cur = get_squred_dist(nbr_vox->coc_glb, cur_glb);
            if(cur_vox->dist_sq > nbr_coc2cur)
            {
                cur_vox->dist_sq =  nbr_coc2cur;
                cur_vox->coc_glb = nbr_vox->coc_glb;
                cur_vox->wave_layer = 1;
                cur_vox->update_ct = map_ct;
                if(!cur_in_q)
                {
                    cur_in_q = true;
                    int aux_id = atomicAdd(q_aux_num,1);
                    q_aux[aux_id] = cur_glb;
                }

            }
        }
    }
}

__device__ __forceinline__
void lower_outside(int3 cur_glb, int index, LocalQueues<int3> &local_q, int *overflow,
                   LocMap &loc_map, HASH_BASE &hashBase, int3* q_aux, int* q_aux_num,
                   int num_dirs, const int3* dirs_D, int gray_shade, int map_ct,
                   bool display_glb_edt, int3* changed_keys, int* changed_cnt)
{
    CrdEqualTo eq;
    if(invalid_src(cur_glb))
        return;

    int3 vb_key = get_VB_key(cur_glb);
    int alloc_id = hashBase.get_alloc_blk_id(vb_key);
    if(alloc_id== -1)
    {
        printf("cur voxel block not found\n");
        assert(false);
    }
    VoxelBlock* Vb= &(hashBase.alloc[alloc_id]);
    GlbVoxel* cur_vox = retrive_vox_D(cur_glb, Vb);

    // streaming
    if(display_glb_edt)
    {
        int  idx_1d = atomicAdd(changed_cnt, 1);
        assert(idx_1d<loc_map._map_volume);
        changed_keys[idx_1d] = vb_key;
    }

    // cut-off dist
    if (cur_vox->dist_sq > loc_map._cutoff_grids_sq)
    {
        return;
    }

    cur_vox->wave_layer =BLACK; // seems redundant
    for(int i=0;i<num_dirs;i++)
    {

        int3 nbr_glb = cur_glb + dirs_D[i];
        int3 nbr_buf = loc_map.glb2loc(nbr_glb);
//        if(!loc_map.is_inside_update_volume(nbr_buf))
        {
            int3 nbr_vb_key = get_VB_key(nbr_glb);
            VoxelBlock* nbr_Vb;
            // judge if nbr and cur are in the same block
            if(eq(nbr_vb_key,vb_key))
            {
                nbr_Vb = Vb;
            }else
            {
                int nbr_alloc_id = hashBase.get_alloc_blk_id(nbr_vb_key);
                if(nbr_alloc_id== -1)
                    continue;

                nbr_Vb= &(hashBase.alloc[nbr_alloc_id]);
            }
            GlbVoxel* nbr_vox = retrive_vox_D(nbr_glb, nbr_Vb);
            // check if nbr is updated by UpdateHashTB()
            if (nbr_vox->vox_type == VOXTYPE_UNKNOWN)
                continue;

            int cur_coc2nbr = get_squred_dist(cur_vox->coc_glb, nbr_glb);
            // previously the vehicle see nothing, so this grid should be propagated anyway
            if(invalid_coc_glb(nbr_vox->coc_glb))
            {

                nbr_vox->dist_sq = cur_coc2nbr;
                nbr_vox->coc_glb = cur_vox->coc_glb;
                local_q.append(index, overflow, nbr_glb);
                continue;
            }


            int origin_dist = atomicMin(&(nbr_vox->dist_sq),cur_coc2nbr);

            // if old dist value is smaller than new dist value: recover the old one
            if (origin_dist<= nbr_vox->dist_sq)
            {
                nbr_vox->dist_sq = origin_dist;
                continue;
            }

            // if cur_coc2nbr in this thread is the minimum
            if(cur_coc2nbr == nbr_vox->dist_sq)
            {

                nbr_vox->coc_glb = cur_vox->coc_glb;
                // only append nbr once in a layer
                int old_color = atomicExch(&(nbr_vox->wave_layer),gray_shade);
                if(old_color == gray_shade && nbr_vox->update_ct == map_ct)
                {
                    continue;
                }
                nbr_vox->update_ct = map_ct;
                local_q.append(index, overflow, nbr_glb);
            }
        }
    }
}

template <class Ktype, class Mtype>
__global__ void
BFS_in_block(waveBase<Ktype> wave, Ktype *q1, Ktype *q2, Ktype *q_aux,
             int num_dirs, const int3* dirs_D, int no_of_nodes, int gray_shade,
             Mtype local_map, HASH_BASE hashBase, int map_ct,
              bool display_glb_edt, int3* changed_keys, int* changed_cnt)
{
    __shared__ LocalQueues<Ktype> local_q;
    __shared__ int prefix_q[NUM_BIN];// store number of elems of each rows

    //next/new wave front
    __shared__ Ktype next_wf[MAX_THREADS_PER_BLOCK];
    __shared__ int  tot_sum;
    if(threadIdx.x == 0)
        tot_sum = 0;//total number of new frontier nodes
    while (1)
    {
        if(threadIdx.x < NUM_BIN){
            local_q.reset(threadIdx.x, blockDim);
        }
        __syncthreads();
        int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
        if(tid<no_of_nodes)
        {
            Ktype crd;  // nodes in frontier
            if(tot_sum == 0)//this is the first BFS level of current kernel call
                crd = q1[tid];
            else
                crd = next_wf[tid];//read the current frontier info from last level's propagation

            if(wave.wave_type[0] == LOWER_OUT)
            {
                lower_outside(crd, threadIdx.x & MOD_OP, local_q, wave.overflow,
                              local_map, hashBase, q_aux, wave.aux_num,
                              num_dirs, dirs_D, gray_shade, map_ct,
                              display_glb_edt, changed_keys, changed_cnt);
            }else if(wave.wave_type[0] == RAISE_OUT)
            {
                raise_outside(crd, threadIdx.x & MOD_OP, local_q, wave.overflow,
                              local_map, hashBase, q_aux, wave.aux_num,
                              num_dirs, dirs_D, gray_shade, map_ct,
                              display_glb_edt, changed_keys, changed_cnt);
            }
        }
        __syncthreads();
        if(threadIdx.x == 0){
            *(wave.f_num) = tot_sum = local_q.size_prefix_sum(prefix_q);
        }
        __syncthreads();
        if(tot_sum == 0)//the new frontier becomes empty; BFS is over
            return;
        if(tot_sum <= MAX_THREADS_PER_BLOCK){
            //the new frontier is still within one-block limit;
            //stay in current kernel
            local_q.concatenate(next_wf, prefix_q);
            __syncthreads();
            no_of_nodes = tot_sum;
            if(threadIdx.x == 0){
                if(gray_shade == GRAY0)
                    gray_shade = GRAY1;
                else
                    gray_shade = GRAY0;
            }
        }else{
            //the new frontier outgrows one-block limit; terminate current kernel
            local_q.concatenate(q2, prefix_q);
            return;
        }
    }

}

template <class Ktype, class Mtype>
__global__ void
BFS_one_layer(waveBase<Ktype> wave, Ktype *q1, Ktype *q2, Ktype *q_aux,
             int num_dirs, const int3* dirs_D, int no_of_nodes, int gray_shade,
             Mtype local_map, HASH_BASE hashBase, int map_ct,
             bool display_glb_edt, int3* changed_keys, int* changed_cnt)
{
    __shared__ LocalQueues<Ktype> local_q;
    __shared__ int prefix_q[NUM_BIN];
    __shared__ int shift;//the number of elementss in the w-queues ahead of current w-queue,

    if(threadIdx.x < NUM_BIN){
        local_q.reset(threadIdx.x, blockDim);
    }
    __syncthreads();

    //first, propagate and add the new frontier elements into w-queues
    int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
    if(tid < no_of_nodes)
    {
        if(wave.wave_type[0] == LOWER_OUT)
        {
            lower_outside(q1[tid], threadIdx.x & MOD_OP, local_q, wave.overflow,
                          local_map, hashBase, q_aux, wave.aux_num,
                          num_dirs, dirs_D, gray_shade, map_ct,
                          display_glb_edt, changed_keys, changed_cnt);
        }else if(wave.wave_type[0] == RAISE_OUT)
        {
            raise_outside(q1[tid], threadIdx.x & MOD_OP, local_q, wave.overflow,
                          local_map, hashBase, q_aux, wave.aux_num,
                          num_dirs, dirs_D, gray_shade, map_ct,
                          display_glb_edt, changed_keys, changed_cnt);
        }
    }
    __syncthreads();
    // Compute size of the output and allocate space in the global queue
    if(threadIdx.x == 0){
        //now calculate the prefix sum
        int tot_sum = local_q.size_prefix_sum(prefix_q);
        //the offset or "shift" of the block-level queue within the
        //grid-level queue is determined by atomic operation
        shift = atomicAdd(wave.f_num,tot_sum);
    }
    __syncthreads();

    //now copy the elements from w-queues into grid-level queues.
    //Note that we have bypassed the copy to/from block-level queues for efficiency reason
    local_q.concatenate(q2 + shift, prefix_q);
}
#endif //SRC_WAVE_CORE_CUH
