#include "pntcld_interfaces.h"
#include <float.h>
#include "ray_cast.h"
#include "par_wave/voxmap_utils.cuh"

namespace PNTCLD_RAYCAST
{

__device__ __forceinline__
bool clearRayLoc(LocMap &loc_map,const int3 &crd, const float &val1, const float &val2, const int &time)
{
    if (loc_map.get_vox_type(crd) != VOXTYPE_OCCUPIED)
    {
        loc_map.atom_add_type_count(crd, -1);
        return true;
    }
    return false;
}


__global__
void getAllocKeys(LocMap loc_map, int3* VB_keys_loc_D, bool for_motion_planner, int rbt_r2_grids)
{
    // get the z and y coordinate of the grid we are about to scan
    int3 loc_crd;
    loc_crd.z = blockIdx.x;
    loc_crd.y = threadIdx.x;

    for (loc_crd.x = 0; loc_crd.x < loc_map._local_size.x; ++loc_crd.x)
    {

        // set grids around as known and free
        if (for_motion_planner)
        {
            int3 crd2center = loc_crd -loc_map._half_shift;
            if(crd2center.x*crd2center.x + crd2center.y*crd2center.y+crd2center.z*crd2center.z <= rbt_r2_grids)
            {
                loc_map.set_vox_count(loc_crd, -1);
            }

        }

        int idx_1d=loc_map.coord2idx_local(loc_crd);

        int count = loc_map.get_vox_count(loc_crd);

        if (count == 0)
        {
            VB_keys_loc_D[idx_1d] = EMPTY_KEY; // vox type is unknown
        }else
        {
            if(count>0)
            {
                loc_map.set_vox_type(loc_crd,VOXTYPE_OCCUPIED);
            }else
            {
                loc_map.set_vox_type(loc_crd,VOXTYPE_FREE);
            }
            int3 glb_crd = loc_map.loc2glb(loc_crd);
            VB_keys_loc_D[idx_1d] = get_VB_key(glb_crd);
        }
    }
}



__global__
void freeLocObs(LocMap loc_map, float3 *pnt_cld, Projection proj, int pnt_sz, int time)
{
    int ring_id = blockIdx.x;
    int scan_id = threadIdx.x;
    int id = threadIdx.x + blockIdx.x *blockDim.x;

    if(id >= pnt_sz)
        return;

    float3 glb_pos = proj.L2G*pnt_cld[id];

    RAY::rayCastLoc(loc_map, proj.origin,  glb_pos, time, 0.707f*loc_map._local_size.x*loc_map._voxel_width, &clearRayLoc);
}


__global__
void registerLocObs(LocMap loc_map, float3 *pnt_cld, Projection proj,  int pnt_sz, int time,
                    int ext_free_num, float3* freeBBX_ll, float3* freeBBX_ur)
{
    int ring_id = blockIdx.x;
    int scan_id = threadIdx.x;
    int id = threadIdx.x + blockIdx.x *blockDim.x;

    if(id >= pnt_sz)
        return;

    float3 glb_pos = proj.L2G*pnt_cld[id];
    bool within_height_limit = glb_pos.z >= loc_map._update_min_h && glb_pos.z <= loc_map._update_max_h;
    bool inside_clear_AABB = false;

    // clean for collision map
    for  (int i=0; i< ext_free_num; i++)
    {
        if(insideAABB( glb_pos, freeBBX_ll[i], freeBBX_ur[i]))
        {
            inside_clear_AABB = true;
            break;
        }
    }

    if (within_height_limit && !inside_clear_AABB)
    {
        int3 glb_crd = loc_map.pos2coord(glb_pos);
        int3 loc_crd = loc_map.glb2loc(glb_crd);

        loc_map.set_vox_type(loc_crd,VOXTYPE_OCCUPIED);
        loc_map.atom_add_type_count(loc_crd,1);
    }
}


void localOGMKernels(LocMap* loc_map, float3 *pnt_cld, Projection proj, PntcldParam param,
                     int3* VB_keys_loc_D, int time, bool for_motion_planner, int rbt_r2_grids,
                     Ext_Obs_Wrapper* ext_obsv)
{
    // Register the point clouds
    registerLocObs<<<param.valid_pnt_count/256+1, 256>>>(*loc_map,pnt_cld,proj,param.valid_pnt_count,time,
                                                         ext_obsv->ext_free_num,
                                                         raw_pointer_cast(&(ext_obsv->freeBBX_ll_D[0])),
                                                         raw_pointer_cast(&(ext_obsv->freeBBX_ur_D[0])));

    // Free the empty areas
    freeLocObs<<<param.valid_pnt_count/256+1, 256>>>(*loc_map,pnt_cld,proj,param.valid_pnt_count,time);

    const int gridSize = loc_map->_local_size.z;
    const int blkSize = loc_map->_local_size.y;
    getAllocKeys<<<gridSize,blkSize>>>(*loc_map,VB_keys_loc_D, for_motion_planner, rbt_r2_grids);

}
}
