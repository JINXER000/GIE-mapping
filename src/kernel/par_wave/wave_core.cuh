

#ifndef SRC_WAVE_CORE_CUH
#define SRC_WAVE_CORE_CUH

#include "par_wave/frontier_wrapper.h"
#include "par_wave/voxmap_utils.cuh"

__device__ unsigned long long int id_atomicMin(unsigned long long int* address, int sq_dist_new, int loc_id_new)
{
    Dist_id new_pair, changing_pair, old_pair;
    new_pair.sq_dist[0] = sq_dist_new;
    new_pair.parent_loc_id[1] = loc_id_new;
    changing_pair.ulong = *address;
    old_pair.ulong = changing_pair.ulong;
    while (changing_pair.sq_dist[0] >  sq_dist_new)
    {
        old_pair.ulong = changing_pair.ulong;
        changing_pair.ulong = atomicCAS(address, changing_pair.ulong,  new_pair.ulong);
    }
    return old_pair.ulong;
}

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
                   LocMap &loc_map, HASH_BASE &hash_base, int3* q_aux, int* q_aux_num,
                   int num_dirs, const int3* dirs_D, int gray_shade, int map_ct,
                   bool display_glb_edt, int3* changed_keys, int* changed_cnt)
{
    CrdEqualTo eq;
    if(invalid_src(cur_glb))
        return;


    int3 vb_key = get_VB_key(cur_glb);
    int alloc_id = hash_base.get_alloc_blk_id(vb_key);
    if(alloc_id <0)
    {
        printf("voxel block not found\n");
        int alloc_id = hash_base.get_alloc_blk_id(vb_key);
        assert(false);
    }
    VoxelBlock* Vb= &(hash_base.alloc[alloc_id]);
    GlbVoxel* cur_vox = retrive_vox_D(cur_glb, Vb);

    if(cur_vox->dist_sq > loc_map._cutoff_grids_sq)
        return;

    // streaming
    if(display_glb_edt)
    {
        int  idx_1d = atomicAdd(changed_cnt, 1);
        assert(idx_1d<loc_map._map_volume);
        changed_keys[idx_1d] = vb_key;
    }

    bool cur_in_q = false;
    int3 valid_local_coc = cur_vox->coc_glb;
    int3 cur_coc_wr = loc_map.glb2wave_range(valid_local_coc);  // must inside loc range

    for(int i=0; i<num_dirs; i++)
    {
        int3 nbr_glb = cur_glb + dirs_D[i];
        if(loc_map.is_inside_local_volume(loc_map.glb2loc(nbr_glb)))
            continue;

        int3 nbr_vb_key = get_VB_key(nbr_glb);
        VoxelBlock* nbr_Vb;
        // judge if nbr and cur are in the same block
        if(eq(nbr_vb_key,vb_key))
        {
            nbr_Vb = Vb;
        }else
        {
            int nbr_alloc_id = hash_base.get_alloc_blk_id(nbr_vb_key);
            if(nbr_alloc_id== -1)
                continue;

            nbr_Vb= &(hash_base.alloc[nbr_alloc_id]);
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
        if(loc_map.is_inside_local_volume(nbr_coc_buf))
        {
            int nbr_coc_new_val = loc_map.g_aux(nbr_coc_buf.x, nbr_coc_buf.y, nbr_coc_buf.z, 0);
            if(nbr_coc_new_val!=0) // disappeared
            {
                int cur_coc2nbr = get_squred_dist(valid_local_coc, nbr_glb);
                nbr_vox->dist_sq = cur_coc2nbr;
                nbr_vox->coc_glb = valid_local_coc;
                nbr_vox->wave_layer = -map_ct;
                nbr_vox->update_ct = -map_ct;

                // update union. cur_coc_wr must be valid since it is inside local range.
                nbr_vox->dist_id_pair.sq_dist[0] = cur_coc2nbr;
                nbr_vox->dist_id_pair.parent_loc_id[1] = loc_map.wr_coc2idx(cur_coc_wr);

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

                // update union in vox
                int3 nbr_coc_wr = loc_map.glb2wave_range(nbr_vox->coc_glb);
                if(!loc_map.is_inside_wave_range(nbr_coc_wr))
                    continue;
                cur_vox->dist_id_pair.sq_dist[0] = nbr_coc2cur;
                cur_vox->dist_id_pair.parent_loc_id[1] = loc_map.wr_coc2idx(nbr_coc_wr);

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
                   LocMap &loc_map, HASH_BASE &hash_base, int3* q_aux, int* q_aux_num,
                   int num_dirs, const int3* dirs_D, int gray_shade, int map_ct,
                   bool display_glb_edt, int3* changed_keys, int* changed_cnt)
{
    volatile int dbg_int=0;
    CrdEqualTo eq;
    if(invalid_src(cur_glb))
        return;

    int3 vb_key = get_VB_key(cur_glb);
    int alloc_id = hash_base.get_alloc_blk_id(vb_key);
    if(alloc_id== -1)
    {
        printf("cur voxel block not found\n");
        assert(false);
    }
    VoxelBlock* Vb= &(hash_base.alloc[alloc_id]);
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

    // propagate coc can be outside the local range! (due to raise out)

    cur_vox->wave_layer =BLACK; // seems redundant

    // update coc and g_aux from parent id
    int cur_coc_id =cur_vox->dist_id_pair.parent_loc_id[1]; // Caution: 0 in wr frame is invalid!
    if(cur_coc_id ==0)
    {
        printf("debug here: invalid cur_coc_id\n");
        assert(false);
    }
    int3 cur_coc_wr = loc_map.id2wr_coc(cur_coc_id);
    int3 cur_coc_glb = loc_map.wave_range2glb(cur_coc_wr);
    cur_vox->coc_glb = cur_coc_glb;
    cur_vox->dist_sq = cur_vox->dist_id_pair.sq_dist[0];



    for(int i=0;i<num_dirs;i++)
    {

        int3 nbr_glb = cur_glb + dirs_D[i];

        int3 nbr_buf = loc_map.glb2loc(nbr_glb);
        if(!loc_map.is_inside_local_volume(nbr_buf))
        {
            int3 nbr_vb_key = get_VB_key(nbr_glb);
            VoxelBlock* nbr_Vb;
            // judge if nbr and cur are in the same block
            if(eq(nbr_vb_key,vb_key))
            {
                nbr_Vb = Vb;
            }else
            {
                int nbr_alloc_id = hash_base.get_alloc_blk_id(nbr_vb_key);
                if(nbr_alloc_id== -1)
                    continue;

                nbr_Vb= &(hash_base.alloc[nbr_alloc_id]);
            }
            GlbVoxel* nbr_vox = retrive_vox_D(nbr_glb, nbr_Vb);
            // check if nbr is updated by UpdateHashTB()
            if (nbr_vox->vox_type == VOXTYPE_UNKNOWN)
                continue;

            int cur_coc2nbr = get_squred_dist(cur_vox->coc_glb, nbr_glb);

            // previously the vehicle see nothing, don't update
            if(invalid_coc_glb(nbr_vox->coc_glb))
            {
                continue;
            }

            Dist_id old_pair;
            old_pair.ulong= id_atomicMin(&(nbr_vox->dist_id_pair.ulong),cur_coc2nbr, cur_coc_id);
            // if cur_coc2nbr in this thread is the minimum
            if(old_pair.sq_dist[0] > cur_coc2nbr )
            {
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
        else{ // inside local map
            int cur_coc2nbr = get_squred_dist(cur_vox->coc_glb, nbr_glb);
            int origin_dist = loc_map.g_aux(nbr_buf.x, nbr_buf.y, nbr_buf.z, 0);

            if(origin_dist>cur_coc2nbr)
            {
                // update union
                int nbr_id = loc_map.id(nbr_buf.x, nbr_buf.y, nbr_buf.z);
                loc_map.dist_in_pair(nbr_id)= cur_coc2nbr;
                loc_map.parent_id_in_pair(nbr_id) = loc_map.wr_coc2idx(cur_coc_wr);
                if(loc_map.wave_layer(nbr_buf.x,nbr_buf.y,nbr_buf.z)==1) // has been in q
                    continue;

                int aux_id = atomicAdd(q_aux_num,1);
                q_aux[aux_id] = nbr_buf;
            }
        }
    }
}


__device__ __forceinline__
void lower_inside(int3 cur_buf, int index, LocalQueues<int3> &local_q, int *overflow,
                  LocMap &loc_map, int num_dirs, const int3* dirs_D, int gray_shade)
{
    if(invalid_src(cur_buf))
        return;

    int cur_id = loc_map.id(cur_buf.x, cur_buf.y, cur_buf.z);
    loc_map.wave_layer(cur_buf.x, cur_buf.y, cur_buf.z)= BLACK;

    int cur_coc_id = loc_map.parent_id_in_pair(cur_id);
    int3 cur_coc_wr = loc_map.id2wr_coc(cur_coc_id);
    int3 cur_coc_buf = loc_map.wave_range2loc(cur_coc_wr);

    for (int i=0; i<num_dirs; i++)
    {
        int3 nbr_buf = cur_buf+ dirs_D[i];
        if(!loc_map.is_inside_local_volume(nbr_buf))
            continue;

        int nbr_id = loc_map.id(nbr_buf.x,nbr_buf.y,nbr_buf.z);
        // if nbr_coc not in loc, it has been visited, but the dist_sq may not be the minimum

        int cur_coc2nbr = get_squred_dist(cur_coc_buf, nbr_buf);

        Dist_id old_pair;
        old_pair.ulong= id_atomicMin(&(loc_map.ulong_in_pair(nbr_id)),cur_coc2nbr, cur_coc_id);

        // if cur_coc2nbr in this thread is the minimum
        if(old_pair.sq_dist[0] > cur_coc2nbr )
        {
            // only append nbr once in a layer
            int old_color = atomicExch(&loc_map.wave_layer(nbr_buf.x,nbr_buf.y,nbr_buf.z),gray_shade);
            if(old_color == gray_shade)
            {
                continue;
            }
            local_q.append(index, overflow, nbr_buf);
        }
    }
}

template <class Ktype, class Mtype>
__global__ void
BFS_in_block(waveBase<Ktype> wave, Ktype *q1, Ktype *q2, Ktype *q_aux,
             int num_dirs, const int3* dirs_D, int no_of_nodes, int gray_shade,
             Mtype local_map, HASH_BASE hash_base, int map_ct,
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

            if(wave.wave_type[0] == LOWER_IN)
            {
                lower_inside(crd, threadIdx.x & MOD_OP, local_q, wave.overflow,
                             local_map, num_dirs, dirs_D, gray_shade);
            }else if(wave.wave_type[0] == LOWER_OUT)
            {
                lower_outside(crd, threadIdx.x & MOD_OP, local_q, wave.overflow,
                              local_map, hash_base, q_aux, wave.aux_num,
                              num_dirs, dirs_D, gray_shade, map_ct,
                              display_glb_edt, changed_keys, changed_cnt);
            }else if(wave.wave_type[0] == RAISE_OUT)
            {
                raise_outside(crd, threadIdx.x & MOD_OP, local_q, wave.overflow,
                              local_map, hash_base, q_aux, wave.aux_num,
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
             Mtype local_map, HASH_BASE hash_base, int map_ct,
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
        if(wave.wave_type[0] == LOWER_IN)
        {
            lower_inside(q1[tid], threadIdx.x & MOD_OP,local_q, wave.overflow,
                         local_map,num_dirs,dirs_D, gray_shade);
        }else if(wave.wave_type[0] == LOWER_OUT)
        {
            lower_outside(q1[tid], threadIdx.x & MOD_OP, local_q, wave.overflow,
                          local_map, hash_base, q_aux, wave.aux_num,
                          num_dirs, dirs_D, gray_shade, map_ct,
                          display_glb_edt, changed_keys, changed_cnt);
        }else if(wave.wave_type[0] == RAISE_OUT)
        {
            raise_outside(q1[tid], threadIdx.x & MOD_OP, local_q, wave.overflow,
                          local_map, hash_base, q_aux, wave.aux_num,
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
