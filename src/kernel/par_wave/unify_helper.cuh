

#ifndef SRC_UNIFY_HELPER_H
#define SRC_UNIFY_HELPER_H

#include "voxmap_utils.cuh"
#include "map_structure/local_batch.h"



__global__
void getUpdatedAddr(int3* stream_VB_keys_D,VoxelBlock** stream_VB_vals_D,
                    HASH_BASE hash_base,int changed_cnt_condense)
{
  const int tid = blockDim.x*blockIdx.x + threadIdx.x;
  if(tid >= changed_cnt_condense)
    return;

  int3 key=stream_VB_keys_D[tid];
  if(invalid_blk_key(key))
    return;

    int alloc_id = hash_base.get_alloc_blk_id(key);
    if(alloc_id== -1)
    {
        printf("cannot find stream block\n");
        assert(false);
    }
    VoxelBlock* Vb= &(hash_base.alloc[alloc_id]);

    stream_VB_vals_D[tid]=Vb;
}


__global__
void updateHashOGMWithPntCld(LocMap loc_map, HASH_BASE hash_base, const int time,
                             bool stream_glb_ogm, int3* stream_VB_keys_D, int* changed_cnt,
                             int ext_obs_num, bool* obs_activated, float3* obsbbx_ll, float3* obsbbx_ur)
{
    int3 loc_crd;
    loc_crd.z = blockIdx.x;
    loc_crd.y = threadIdx.x;

    int3 glb_crd;
    for (loc_crd.x = 0; loc_crd.x < loc_map._local_size.x; ++loc_crd.x) {

        // reset observation of one scan
        int count = loc_map.get_vox_count(loc_crd);
        loc_map.set_vox_count(loc_crd,0);
        loc_map.set_vox_type(loc_crd,VOXTYPE_UNKNOWN);

        // get voxel in hash
        glb_crd= loc_map.loc2glb(loc_crd);
        int3 vb_key = get_VB_key(glb_crd);
        int alloc_id = hash_base.get_alloc_blk_id(vb_key);

        // vb is unobserved and unallocated ever
        if(alloc_id== -1)
        {
            loc_map.set_vox_glb_type(loc_crd,VOXTYPE_UNKNOWN);
            continue;
        }
        VoxelBlock* Vb= &(hash_base.alloc[alloc_id]);
        GlbVoxel* cur_vox = retrive_vox_D(glb_crd, Vb);

        char old_glb_type = cur_vox->vox_type;

        // if outside outbbx and inside obs_bbx, set as occupied
        float3 glb_pos=loc_map.coord2pos(glb_crd);
        bool occ_flag = false;


        if(obs_activated[0] && !insideAABB(glb_pos, obsbbx_ll[0], obsbbx_ur[0]))
        {
            occ_flag = true;
        }else
        {
            for(int i=1; i<ext_obs_num; i++)
            {
                if(obs_activated[i] && insideAABB(glb_pos, obsbbx_ll[i], obsbbx_ur[i]))
                {
                    occ_flag = true;
                    break;
                }
            }
        }

        // update observed occ val from sensor/ext observ
        if (count > 0 || occ_flag)
        {
            set_hashvoxel_occ_val(cur_vox,250.f, 1.f, loc_map._occu_thresh, time);
        }
        else if (count < 0)
        {
            float pbty = min(1.f,static_cast<float>(-count)/10.f);
            set_hashvoxel_occ_val(cur_vox,0.f,pbty,loc_map._occu_thresh, time);
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
void updateHashOGMWithSensor(LocMap loc_map, HASH_BASE hash_base, const int time,
                             bool stream_glb_ogm, int3* stream_VB_keys_D, int* changed_cnt,
                             int ext_obs_num, bool* obs_activated, float3* obsbbx_ll, float3* obsbbx_ur)
{
    int3 loc_crd;
    loc_crd.z = blockIdx.x;
    loc_crd.y = threadIdx.x;

    int3 glb_crd;
    for (loc_crd.x = 0; loc_crd.x < loc_map._local_size.x; ++loc_crd.x) {

        char new_vox_type = loc_map.get_vox_type(loc_crd);
        // reset observation of one scan
        loc_map.set_vox_type(loc_crd,VOXTYPE_UNKNOWN);

        // get voxel in hash
        glb_crd= loc_map.loc2glb(loc_crd);
        int3 vb_key = get_VB_key(glb_crd);
        int alloc_id = hash_base.get_alloc_blk_id(vb_key);

        // vb is unobserved and unallocated ever
        if(alloc_id== -1)
        {
            loc_map.set_vox_glb_type(loc_crd,VOXTYPE_UNKNOWN);
            continue;
        }

        // if outside outbbx and inside obs_bbx, set as occupied
        float3 glb_pos=loc_map.coord2pos(glb_crd);
        bool occ_flag = false;
        if(obs_activated[0] && !insideAABB(glb_pos, obsbbx_ll[0], obsbbx_ur[0]))
        {
            occ_flag = true;
        }else
        {
            for(int i=1; i<ext_obs_num; i++)
            {
                if(obs_activated[i] && insideAABB(glb_pos, obsbbx_ll[i], obsbbx_ur[i]))
                {
                    occ_flag = true;
                    break;
                }
            }
        }

        VoxelBlock* Vb= &(hash_base.alloc[alloc_id]);
        GlbVoxel* cur_vox = retrive_vox_D(glb_crd, Vb);

        char old_glb_type = cur_vox->vox_type;
        // update observed occ val

        if (new_vox_type == VOXTYPE_OCCUPIED || occ_flag)
        {
            set_hashvoxel_occ_val(cur_vox,250.f, 0.8f, loc_map._occu_thresh, time);
        }
        else if (new_vox_type == VOXTYPE_FREE)
        {
            set_hashvoxel_occ_val(cur_vox,0.f,0.5f,loc_map._occu_thresh, time);
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
void MarkLimitedObserve(LocMap loc_map,HASH_BASE hash_base,
                     bool display_glb_edt, int3* stream_VB_keys_D, int* changed_cnt) {
    int3 c;
    c.z = blockIdx.x;
    c.x = threadIdx.x;

    char cur_vox_type;
    int dist_sq_new, dist_sq_old;
    int3 coc_new, cur_coc_glb_old;
    for (c.y = 0; c.y < loc_map._local_size.y; c.y++) {

        cur_vox_type = loc_map.get_vox_glb_type(c);
        int3 cur_glb = loc_map.loc2glb(c);


        if (cur_vox_type == VOXTYPE_UNKNOWN)
            continue;
        int cur_id = loc_map.id(c.x, c.y, c.z);

        coc_new = loc_map.get_coc_viaID(c.x, c.y, c.z, 0, true);
        dist_sq_new = loc_map.g_aux(c.x, c.y, c.z, 0);


        if (invalid_coc_buf(coc_new, loc_map._max_width))
        {
            // see nothing, but can be updated by limited observation process later
            loc_map.dist_in_pair(cur_id)= EMPTY_VALUE; // set dist invalid, since every bit in parent_id is meaningful
            loc_map.parent_id_in_pair(cur_id) = 0xffffffff;
            loc_map.g_aux(c.x, c.y, c.z, 0) = EMPTY_VALUE; // instead of recognize it as invalid, it will be regarded as unseen
        }

        // update hash table
        int3 vb_key = get_VB_key(cur_glb);

        int alloc_id = hash_base.get_alloc_blk_id(vb_key);
        if (alloc_id == -1) {
            printf("voxel block of %d, %d, %d not found\n", vb_key.x, vb_key.y, vb_key.z);
            assert(false);
            continue;
        }

        VoxelBlock *Vb = &(hash_base.alloc[alloc_id]);
        GlbVoxel *cur_vox = retrive_vox_D(cur_glb, Vb);

        dist_sq_old = cur_vox->dist_sq;
        cur_coc_glb_old = cur_vox->coc_glb;
        int3 cur_coc_buf_old = loc_map.glb2loc(cur_coc_glb_old);
        bool old_coc_inloc = loc_map.is_inside_local_volume(cur_coc_buf_old); // EMPTY_VALUE will be outside
        bool limited_observe = dist_sq_new > dist_sq_old && !old_coc_inloc;
        if (limited_observe) {
            // limited observation: update batch
            coc_new = cur_coc_buf_old;
            loc_map.g_aux(c.x, c.y, c.z, 0) = dist_sq_old;
        }

        // copy g and coc to id_dist pair. Check coc inside update range
        int3 cur_coc_wr = loc_map.loc2wave_range(coc_new);
        if(!loc_map.is_inside_wave_range(cur_coc_wr))
        {
            loc_map.dist_in_pair(cur_id)= EMPTY_VALUE; // set dist invalid, since every bit in parent_id is meaningful
            loc_map.g_aux(c.x, c.y, c.z, 0) = EMPTY_VALUE; // instead of recognize it as invalid, it will be regarded as unseen
        }else
        {
            loc_map.dist_in_pair(cur_id)= loc_map.g_aux(c.x, c.y, c.z, 0);
            loc_map.parent_id_in_pair(cur_id) = loc_map.wr_coc2idx(cur_coc_wr);
        }
        // copy new batch value to read-only backups (left)
        loc_map.g(c.x, c.y, c.z, 0) = loc_map.g_aux(c.x, c.y, c.z, 0);
        loc_map.coc_idx(c.x, c.y, c.z, 0) = loc_map.parent_id_in_pair(cur_id); // actually wr, not loc

    }
}

__global__ 
void obtainFrontiers(LocMap loc_map, HASH_BASE hash_base,int3* frontierA, int3* frontierB,  int3* frontierC,
                     int* fa_num, int* fb_num, int* fc_num,
                     int num_dirs, int3* dirs_D,int map_ct) {
    int3 c;
    c.z = blockIdx.x;
    c.x = threadIdx.x;

    for (c.y = 0; c.y < loc_map._local_size.y; c.y++) {
        loc_map.wave_layer(c.x,c.y,c.z) = EMPTY_VALUE;
        char cur_vox_type = loc_map.get_vox_glb_type(c);
        if(cur_vox_type==VOXTYPE_UNKNOWN)
            continue;

        int3 cur_glb = loc_map.loc2glb(c);
        int cur_id = loc_map.id(c.x, c.y, c.z);
        int3 cur_coc_wr = loc_map.get_coc_viaID(c.x, c.y, c.z, 0, false);  // !It is actually the left_value of cur wave coc
        int3 cur_coc_buf = loc_map.wave_range2loc(cur_coc_wr);
        int3 cur_coc_glb = loc_map.loc2glb(cur_coc_buf);


        int cur_dist_sq = loc_map.g(c.x, c.y, c.z, 0);

        bool cur_coc_in_loc = loc_map.is_inside_local_volume(cur_coc_buf);
        // if false, then it is due to limited observation (or see nothing). no need to extract frontier
        if(!cur_coc_in_loc)
        {
            continue;
        }
        bool cur_in_q = false; // if true, then cur_glb has been pushed to a frontier

        bool nbr_has_unknown = false;
        for (int i = 0; i < num_dirs; i++) {
            int3 nbr_buf = c + dirs_D[i];

            bool is_nbr_local = loc_map.is_inside_local_volume(nbr_buf);
            if (is_nbr_local)
            {
                // if nbr is ever observed, then it may be a lower_in frontier
                char nbr_buf_type = loc_map.get_vox_glb_type(nbr_buf);
                if(nbr_buf_type == VOXTYPE_UNKNOWN)
                {
                    nbr_has_unknown = true;
                    continue;
                }

                // if cur coc in loc and nbr coc not in loc, compare dist
                int nbr_dist_sq = loc_map.g(nbr_buf.x,nbr_buf.y,nbr_buf.z,0);
                int3 nbr_coc_wr = loc_map.get_coc_viaID(nbr_buf.x,nbr_buf.y,nbr_buf.z, 0, false);  // left value of nbr wave coc
                int3 nbr_coc_buf = loc_map.wave_range2loc(nbr_coc_wr);

                bool is_nbr_coc_valid = loc_map.is_inside_wave_range(nbr_coc_wr);
                bool is_nbr_coc_in_loc = loc_map.is_inside_local_volume(nbr_coc_buf);
                if(!is_nbr_coc_in_loc && is_nbr_coc_valid)
                {
                    int nbr_coc2cur = get_squred_dist(nbr_coc_buf, c);
                    if(nbr_coc2cur < cur_dist_sq)
                    {
                        // update union
                        loc_map.dist_in_pair(cur_id)= nbr_coc2cur;
                        loc_map.parent_id_in_pair(cur_id) = loc_map.wr_coc2idx(nbr_coc_wr);
                        if(!cur_in_q)
                        {
                            cur_in_q = true;
                            loc_map.wave_layer(cur_id) = 1;

                            int fc_id = atomicAdd(fc_num, 1);
                            frontierC[fc_id] = c;
                        }
                    }
                }
            }else
            {
                // obtain nbr voxel
                int3 nbr_glb = loc_map.loc2glb(nbr_buf);
                int3 vb_key = get_VB_key(nbr_glb);
                int alloc_id = hash_base.get_alloc_blk_id(vb_key);
                if (alloc_id == -1) {
                    nbr_has_unknown = true;
                    continue;
                }
                VoxelBlock *Vb = &(hash_base.alloc[alloc_id]);
                GlbVoxel *nbr_vox = retrive_vox_D(nbr_glb, Vb);

                if(nbr_vox->vox_type==VOXTYPE_UNKNOWN)
                {
                    nbr_has_unknown = true;
                    continue;
                }

                int nbr_dist_sq = nbr_vox->dist_sq;
                if (invalid_dist_glb(nbr_dist_sq))// maybe the last frame sees nothing
                {
                    continue;
                }
                int3 nbr_coc_glb = nbr_vox->coc_glb;
                if (invalid_coc_glb(nbr_coc_glb))
                {
                    continue;
                }

                int3 nbr_coc_wr = loc_map.glb2wave_range(nbr_coc_glb);
                bool is_nbr_coc_valid = loc_map.is_inside_wave_range(nbr_coc_wr);
                int3 nbr_coc_buf = loc_map.glb2loc(nbr_coc_glb);
                bool is_nbr_coc_local =loc_map.is_inside_local_volume(nbr_coc_buf);

                //if cur coc in loc and nbr coc not in loc, compare dist
                if(!is_nbr_coc_local && is_nbr_coc_valid)
                {
                    int nbr_coc2cur = get_squred_dist(nbr_coc_buf, c);
                    if(nbr_coc2cur < cur_dist_sq)
                    {
                        // update union
                        loc_map.dist_in_pair(cur_id)= nbr_coc2cur;
                        loc_map.parent_id_in_pair(cur_id) = loc_map.wr_coc2idx(nbr_coc_wr);
                        if(!cur_in_q)
                        {
                            cur_in_q = true;
                            loc_map.wave_layer(cur_id) = 1;

                            int fc_id = atomicAdd(fc_num, 1);
                            frontierC[fc_id] = c;
                        }
                    }
                }

                if(loc_map._fast_mode)
                    continue;

                int cur_coc2nbr = get_squred_dist(nbr_buf,cur_coc_buf);

                // lower out
                if(cur_coc2nbr < nbr_dist_sq)
                {
                    nbr_vox->wave_layer = 1;
                    nbr_vox->update_ct = map_ct;

                    // update union in vox
                    nbr_vox->dist_id_pair.sq_dist[0] = cur_coc2nbr;
                    nbr_vox->dist_id_pair.parent_loc_id[1] = loc_map.wr_coc2idx(cur_coc_wr);


                    int fb_id = atomicAdd(fb_num, 1);
                    frontierB[fb_id] = nbr_glb;

                }
                else if(cur_coc2nbr>nbr_dist_sq && is_nbr_coc_local) // raise_out
                {
                    char nbr_coc_type = loc_map.get_vox_glb_type(nbr_coc_buf);
                    if(nbr_coc_type != VOXTYPE_OCCUPIED)
                    {
                        nbr_vox->dist_sq = cur_coc2nbr;
                        nbr_vox->coc_glb = cur_coc_glb;
                        nbr_vox->wave_layer = -map_ct;

                        // update union in vox
                        nbr_vox->dist_id_pair.sq_dist[0] = cur_coc2nbr;
                        nbr_vox->dist_id_pair.parent_loc_id[1] = loc_map.wr_coc2idx(cur_coc_wr);

                        int fa_id = atomicAdd(fa_num, 1);
                        frontierA[fa_id] = nbr_glb;
                    }
                }
            }
        }

        if(cur_vox_type== VOXTYPE_FREE  && nbr_has_unknown)
        {
            loc_map.set_vox_glb_type(c, VOXTYPE_FNT);
        }
    }
}

__global__
void UpdateHashBatch(LocMap loc_map,HASH_BASE hash_base,
                  bool display_glb_edt, int3* stream_VB_keys_D, int* changed_cnt)
{
    int3 c;
    c.z = blockIdx.x;
    c.x = threadIdx.x;

    for (c.y=0;c.y< loc_map._local_size.y;c.y++)
    {
        int cur_id = loc_map.id(c.x, c.y, c.z);

        char cur_vox_type = loc_map.get_vox_glb_type(c);
        int3 cur_glb = loc_map.loc2glb(c);

        if (cur_vox_type == VOXTYPE_UNKNOWN)
            continue;

        // exception
        if(loc_map.dist_in_pair(cur_id) == EMPTY_VALUE)
        {
            // see nothing, update
            if(loc_map.parent_id_in_pair(cur_id) == 0xffffffff)
            {
                loc_map.edt_gpu_out(c.x, c.y, c.z) = loc_map._max_loc_dist_sq;
            }// else coc outside the wave range, don't update
            continue;
        }


        // update hash table
        int3 vb_key = get_VB_key(cur_glb);

        int alloc_id = hash_base.get_alloc_blk_id(vb_key);
        if(alloc_id== -1)
        {
            printf("voxel block of %d, %d, %d not found\n", vb_key.x, vb_key.y, vb_key.z);
            assert(false);
            continue;
        }

        VoxelBlock* Vb= &(hash_base.alloc[alloc_id]);
        GlbVoxel* cur_vox = retrive_vox_D(cur_glb, Vb);

        int cur_coc_id = loc_map.parent_id_in_pair(cur_id);
        int3 cur_coc_wr = loc_map.id2wr_coc(cur_coc_id);
        int dist_sq_new = loc_map.dist_in_pair(cur_id);
        int dist_sq_old = cur_vox->dist_sq;

        cur_vox->coc_glb = loc_map.wave_range2glb(cur_coc_wr);
        cur_vox->dist_sq = dist_sq_new;
        loc_map.edt_gpu_out(c.x, c.y, c.z) = sqrtf(dist_sq_new);

        // update  union
        cur_vox->dist_id_pair.ulong = loc_map.ulong_in_pair(cur_id);

        if(cur_vox_type == VOXTYPE_FNT)
        {
            cur_vox->vox_type= VOXTYPE_FNT;
        }


        if(display_glb_edt && dist_sq_new!= dist_sq_old)
        {
            int idx_1d = atomicAdd(changed_cnt, 1);
            if(idx_1d>=loc_map._map_volume)
            {
                printf("--Caution: Stream capacity not enough!\n");
                assert(false);
                continue;
            }
            stream_VB_keys_D[idx_1d] = vb_key;
        }

    }
}
#endif //SRC_UNIFY_HELPER_H
