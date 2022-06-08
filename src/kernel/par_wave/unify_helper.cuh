
#ifndef SRC_UNIFY_HELPER_H
#define SRC_UNIFY_HELPER_H

#include "voxmap_utils.cuh"
#include "map_structure/local_batch.h"


__global__ __forceinline__
void getUpdatedAddr(int3* stream_VB_keys_D,VoxelBlock** stream_VB_vals_D,
                    HASH_BASE hashBase,int changed_cnt_condense)
{
  const int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if(tid >= changed_cnt_condense)
    return;

  int3 key=stream_VB_keys_D[tid];
  if(invalid_blk_key(key))
    return;

    int alloc_id = hashBase.get_alloc_blk_id(key);
    if(alloc_id== -1)
    {
        printf("cannot find stream block\n");
        assert(false);
    }
    VoxelBlock* Vb= &(hashBase.alloc[alloc_id]);

    stream_VB_vals_D[tid]=Vb;
}


__global__ __forceinline__
void updateHashOGMWithPntCld(LocMap loc_map, HASH_BASE hashBase, const int time,
                             bool stream_glb_ogm, int3* stream_VB_keys_D, int* changed_cnt)
{
    int3 loc_crd;
    loc_crd.z = blockIdx.x;
    loc_crd.y = threadIdx.x;

    int3 glb_crd;
    for (loc_crd.x = 0; loc_crd.x < loc_map._update_size.x; ++loc_crd.x) {

        // reset observation of one scan
        int count = loc_map.get_vox_count(loc_crd);
        loc_map.set_vox_count(loc_crd,0);
        loc_map.set_vox_type(loc_crd,VOXTYPE_UNKNOWN);

        // get voxel in hash
        glb_crd= loc_map.loc2glb(loc_crd);
        int3 vb_key = get_VB_key(glb_crd);
        int alloc_id = hashBase.get_alloc_blk_id(vb_key);

        // vb is unobserved and unallocated ever
        if(alloc_id== -1)
        {
            loc_map.set_vox_glb_type(loc_crd,VOXTYPE_UNKNOWN);
            continue;
        }
        VoxelBlock* Vb= &(hashBase.alloc[alloc_id]);
        GlbVoxel* cur_vox = retrive_vox_D(glb_crd, Vb);

        char old_glb_type = cur_vox->vox_type;
        // update observed occ val
        if(count!=0)
        {
            if (count > 0)
            {
                set_hashvoxel_occ_val(cur_vox,250.f, 1.f, loc_map._occu_thresh, time);
            }

            if (count < 0)
            {
                float pbty = min(1.f,static_cast<float>(-count)/10.f);
                set_hashvoxel_occ_val(cur_vox,0.f,pbty,loc_map._occu_thresh, time);
            }
        }

        // set val back for local edt

        loc_map.set_vox_glb_type(loc_crd, cur_vox->vox_type);

        if(!stream_glb_ogm)
            continue;
        if(cur_vox->vox_type!= old_glb_type)
        {
            int idx_1d = atomicAdd(changed_cnt, 1);
            if(idx_1d>=loc_map._map_volume)
            {
                printf("--Caution: Stream capacity not enough!\n");
                assert(false);
            }
            stream_VB_keys_D[idx_1d] = vb_key;
        }
    }
}

__global__ __forceinline__
void updateHashOGMWithSensor(LocMap loc_map, HASH_BASE hashBase, const int time,
                             bool stream_glb_ogm, int3* stream_VB_keys_D, int* changed_cnt)
{
    int3 loc_crd;
    loc_crd.z = blockIdx.x;
    loc_crd.y = threadIdx.x;

    int3 glb_crd;
    for (loc_crd.x = 0; loc_crd.x < loc_map._update_size.x; ++loc_crd.x) {

        char new_vox_type = loc_map.get_vox_type(loc_crd);
        // reset observation of one scan
        loc_map.set_vox_type(loc_crd,VOXTYPE_UNKNOWN);

        // get voxel in hash
        glb_crd= loc_map.loc2glb(loc_crd);
        int3 vb_key = get_VB_key(glb_crd);
        int alloc_id = hashBase.get_alloc_blk_id(vb_key);

        // vb is unobserved and unallocated ever
        if(alloc_id== -1)
        {
            loc_map.set_vox_glb_type(loc_crd,VOXTYPE_UNKNOWN);
            continue;
        }
        VoxelBlock* Vb= &(hashBase.alloc[alloc_id]);
        GlbVoxel* cur_vox = retrive_vox_D(glb_crd, Vb);

        char old_glb_type = cur_vox->vox_type;
        // update observed occ val
        if(new_vox_type!=VOXTYPE_UNKNOWN)
        {
            if (new_vox_type == VOXTYPE_OCCUPIED)
            {
                set_hashvoxel_occ_val(cur_vox,250.f, 0.4f, loc_map._occu_thresh, time);
            }

            if (new_vox_type == VOXTYPE_FREE)
            {
                set_hashvoxel_occ_val(cur_vox,0.f,0.5f,loc_map._occu_thresh, time);
            }
        }

        // set val back for local edt

        loc_map.set_vox_glb_type(loc_crd, cur_vox->vox_type);

        if(!stream_glb_ogm)
            continue;
        if(cur_vox->vox_type!= old_glb_type)
        {
            int idx_1d = atomicAdd(changed_cnt, 1);
            if(idx_1d>=loc_map._map_volume)
            {
                printf("--Caution: Stream capacity not enough!\n");
                assert(false);
            }
            stream_VB_keys_D[idx_1d] = vb_key;
        }
    }
}

__global__
void batchEDTUnify(LocMap loc_map,HASH_BASE hashBase,int3* frontierA, int3* frontierB,
                   int num_dirs, int3* dirs_D,int map_ct,
                     bool display_glb_edt, int3* stream_VB_keys_D, int* changed_cnt)
{
    CrdEqualTo eq;
    int3 c;
    c.z = blockIdx.x;
    c.x = threadIdx.x;

    for (c.y=0;c.y<loc_map._update_size.y;c.y++)
    {
        int cur_vox_type = loc_map.get_vox_glb_type(c);
        if(cur_vox_type==VOXTYPE_UNKNOWN)
            continue;

        bool see_nothing = false;
        // check buf coc
        int3 coc_new = loc_map.coc(c.x, c.y, c.z);
        int dist_sq_new = loc_map.g_aux(c.x,c.y,c.z,0);
        if(invalid_coc_buf(coc_new))
        {
            if(dist_sq_new < loc_map._max_loc_dist_sq)
                assert(false);
            else{
                see_nothing = true;
            }
        }


        // update hash table
        int3 cur_glb= loc_map.loc2glb(c);
        int3 vb_key = get_VB_key(cur_glb);

        int alloc_id = hashBase.get_alloc_blk_id(vb_key);
        if(alloc_id== -1)
        {
            printf("voxel block of %d, %d, %d not found\n", vb_key.x, vb_key.y, vb_key.z);
            assert(false);
        }

        VoxelBlock* Vb= &(hashBase.alloc[alloc_id]);
        GlbVoxel* cur_vox = retrive_vox_D(cur_glb, Vb);

        bool stream_this_vox = false;

        int3 cur_coc_glb_new = loc_map.loc2glb(coc_new);
        int dist_sq_old = cur_vox->dist_sq;
        int3 cur_coc_glb_old = cur_vox->coc_glb;
        int3 cur_coc_buf_old = loc_map.glb2loc(cur_coc_glb_old);
        bool old_coc_inloc = loc_map.is_inside_update_volume(cur_coc_buf_old); // EMPTY_VALUE will be outside

        // special case: old coc disappear?
        int old_coc_stat = VOXTYPE_OCCUPIED;
        if(old_coc_inloc)
        {
            old_coc_stat = loc_map.get_vox_glb_type(cur_coc_buf_old);
        }
        // judge conditions
        int bdr_face = loc_map.decide_bdr_face(c);
        bool old_coc_gone_moredist = old_coc_inloc && old_coc_stat == VOXTYPE_FREE && dist_sq_new > dist_sq_old;
        bool limited_observe = dist_sq_new > dist_sq_old && !old_coc_inloc;

        if(old_coc_gone_moredist && bdr_face>0)
        {
            stream_this_vox = true;
        }
        else if(limited_observe )
        {
            // limited observation: update batch
            loc_map.coc(c.x,c.y,c.z) = cur_coc_buf_old;
            loc_map.g_aux(c.x,c.y,c.z,0) = dist_sq_old;
            loc_map.edt_gpu_out(c.x,c.y,c.z)=sqrtf(dist_sq_old);

            // if see nothing: 52900 > 30 && old_coc_inloc == false, don't update hash
        }else
        {
            // pop up or disappear in loc: update hash
            cur_vox->dist_sq = dist_sq_new;
            if(!see_nothing)
            cur_vox->coc_glb = cur_coc_glb_new;
            loc_map.edt_gpu_out(c.x,c.y,c.z)=sqrtf(dist_sq_new);

            if(display_glb_edt && dist_sq_new!= dist_sq_old)
            {
                stream_this_vox = true;
            }
        }

        if(see_nothing)
        {
            continue;
        }
        int3 cur_coc_buf_updated = loc_map.coc(c.x,c.y,c.z);

        // extract frontiers!
        bool batch_acc_flag = true;  // assume batch edt is accurate when old_coc_gone_moredist == true
        bool nbr_has_unknown = false;
        for (int i = 0; i < num_dirs; i++) {
            int3 nbr_buf = c + dirs_D[i];

            bool is_nbr_local = loc_map.is_inside_update_volume(nbr_buf);
            if (is_nbr_local) // todo: merge 2 kernels, get cur_vox
            {
                int nbr_buf_type = loc_map.get_vox_glb_type(nbr_buf);
                if(nbr_buf_type == VOXTYPE_UNKNOWN)
                    nbr_has_unknown = true;
            }else
            {
                // obtain nbr voxel
                int3 nbr_glb = loc_map.loc2glb(nbr_buf);
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
                GlbVoxel *nbr_vox = retrive_vox_D(nbr_glb, nbr_Vb);

                if(nbr_vox->vox_type==VOXTYPE_UNKNOWN)
                {
                    nbr_has_unknown = true;
                    continue;
                }

                int nbr_dist_sq = nbr_vox->dist_sq;
                if (invalid_dist_glb(nbr_dist_sq))
                {
                    continue;
                }
                int3 nbr_coc_glb = nbr_vox->coc_glb;
                if (invalid_coc_glb(nbr_coc_glb)) //nbr could be seeing nothing
                {
                    continue;
                }

                int3 nbr_coc_buf = loc_map.glb2loc(nbr_coc_glb);
                bool is_nbr_coc_local =loc_map.is_inside_update_volume(nbr_coc_buf);

                // lower_in.
                if(!is_nbr_coc_local && old_coc_gone_moredist && bdr_face>0)
                {
                    int nbr_coc2cur = get_squred_dist(cur_glb, nbr_coc_glb);
                    int batch_edt_val = loc_map.g_aux(c.x,c.y,c.z,0);
                    if(nbr_coc2cur < batch_edt_val)
                    {
                        // limited observe: treat as src of frontierB
                        loc_map.g_aux(c.x,c.y,c.z,0) = nbr_coc2cur;
                        loc_map.edt_gpu_out(c.x,c.y,c.z)=sqrtf(nbr_coc2cur);
                        cur_vox->dist_sq = nbr_coc2cur;
                        cur_vox->coc_glb = nbr_vox->coc_glb;
                        cur_vox->wave_layer = 1;
                        cur_vox->update_ct = map_ct;

                        int bdr_idx = loc_map.get_bdr_idx(i,c);
                        frontierB[bdr_idx] = cur_glb;
                        batch_acc_flag = false;


                    }
                    // else: batch edt may be accurate (need to check other nbrs)
                }

                int cur_coc2nbr = get_squred_dist(nbr_buf,cur_coc_buf_updated);
                // consider remove this
                if(cur_coc2nbr > loc_map._cutoff_grids_sq)
                    continue;
                // lower out
                if(!old_coc_gone_moredist && cur_coc2nbr < nbr_dist_sq ) // && dist_sq_new < dist_sq_old
                {
                     nbr_vox->dist_sq = cur_coc2nbr;
                    nbr_vox->coc_glb = cur_vox->coc_glb;
                    nbr_vox->wave_layer = 1;
                    nbr_vox->update_ct = map_ct;

                    int bdr_idx = loc_map.get_bdr_idx(i,c);
                    frontierB[bdr_idx] = nbr_glb;


                }else if(cur_coc2nbr>nbr_dist_sq && is_nbr_coc_local) // raise_out
                {
                    int nbr_coc_type = loc_map.get_vox_glb_type(nbr_coc_buf);
                    if(nbr_coc_type != VOXTYPE_OCCUPIED)
                    {
                        nbr_vox->dist_sq = cur_coc2nbr;
                        nbr_vox->coc_glb = cur_vox->coc_glb;
                        nbr_vox->wave_layer = -map_ct;
//                        nbr_vox->update_ct = -map_ct;

                        int bdr_idx = loc_map.get_bdr_idx(i,c);
                        frontierA[bdr_idx] = nbr_glb;

                    }
                }
            }
        }

        if(old_coc_gone_moredist && bdr_face>0 && batch_acc_flag)
        {
            // update hash
            cur_vox->dist_sq = dist_sq_new;
            cur_vox->coc_glb = cur_coc_glb_new;
            loc_map.edt_gpu_out(c.x,c.y,c.z)=sqrtf(dist_sq_new);
        }
        if(stream_this_vox ==true)
        {
            int idx_1d = atomicAdd(changed_cnt, 1);
            if(idx_1d>=loc_map._map_volume)
            {
                printf("--Caution: Stream capacity not enough!\n");
                assert(false);
            }
            stream_VB_keys_D[idx_1d] = vb_key;
        }
        // frontiers for exploration
       if(cur_vox->vox_type == VOXTYPE_FREE  && nbr_has_unknown)
       {
           cur_vox->vox_type= VOXTYPE_FNT;
       }
    }
}

#endif //SRC_UNIFY_HELPER_H
