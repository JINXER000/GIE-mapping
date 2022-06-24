#ifndef RAY_CAST_H
#define RAY_CAST_H

#include "cuda_toolkit/cuda_macro.h"
#include "cuda_toolkit/projection.h"
#include "cuda_toolkit/occupancy/point_cloud/pntcld_param.h"
#include <float.h>
#include "map_structure/local_batch.h"

namespace RAY
{
__device__ __forceinline__
int& at(int3& val, const int& id)
{
    switch (id)
    {
    case 0: return val.x;
    case 1: return val.y;
    case 2: return val.z;
    default:  assert(0);
    }

    return val.x;
}

__device__ __forceinline__
float& at(float3& val, const int& id)
{
    switch (id)
    {
    case 0: return val.x;
    case 1: return val.y;
    case 2: return val.z;
    default:  assert(0);
    }

    return val.x;
}

__device__ __forceinline__
float at(const float3& val, const int& id)
{
    switch (id)
    {
    case 0: return val.x;
    case 1: return val.y;
    case 2: return val.z;
    default:  assert(0);
    }

    return val.x;
}


typedef bool (*ray_loc_operator)(LocMap &loc_map,const int3 &crd, const float &val1, const float &val2, const int &time);
__device__ __forceinline__
void rayCastLoc(LocMap &loc_map, const float3 &p0, const float3 &p1,  int time, float max_length, ray_loc_operator opr)
{
    int3 p0Index = loc_map.pos2coord(p0);
    int3 p1Index = loc_map.pos2coord(p1);

    opr(loc_map,loc_map.glb2loc(p0Index),-1,0,time);
    // same grid, we are done
    if(p0Index.x == p1Index.x && p0Index.y == p1Index.y && p0Index.z == p1Index.z)
    {
        return;
    }
    // Initialization phase ------------------------
    float3 direction = p1 - p0;
    float len = length(direction);
    direction = direction / len; //normalize the vector

    int    step[3];
    float tMax[3];
    float tDelta[3];

    int3 currIndex = p0Index;
    for (unsigned int i=0; i<3; i++)
    {
        // compute step direction
        if(at(direction, i) > 0.0f) step[i] = 1;
        else if (at(direction, i) < 0.0f) step[i] = -1;
        else step[i] = 0;

        // compute tMax, tDelta
        if(step[i] != 0)
        {
            float voxelBorder = float(at(currIndex, i)) * loc_map._voxel_width +
                                float(step[i]) * loc_map._voxel_width*0.5f;
            tMax[i] = (voxelBorder - at(p0, i))/at(direction, i);
            tDelta[i] = loc_map._voxel_width / fabs(at(direction, i));
        }
        else
        {
            tMax[i] =  FLT_MAX;
            tDelta[i] = FLT_MAX;
        }
    }

    // Incremental phase -----------------------------------
    bool done = false;
    while (!done)
    {
        unsigned int dim;

        // find minimum tMax;
        if (tMax[0] < tMax[1]){
            if (tMax[0] < tMax[2]) dim = 0;
            else                   dim = 2;
        }
        else {
            if (tMax[1] < tMax[2]) dim = 1;
            else                   dim = 2;
        }

        // advance in drection "dim"
        at(currIndex, dim) += step[dim];
        tMax[dim] += tDelta[dim];

        if(!opr(loc_map, loc_map.glb2loc(currIndex), -1, 0, time))
        {
            done = true;
            break;
        }

        // reached endpoint?
        if(currIndex.x == p1Index.x && currIndex.y == p1Index.y && currIndex.z == p1Index.z)
        {
            done = true;
            break;
        }
        else
        {
            float dist_from_origin = min(min(tMax[0], tMax[1]), tMax[2]);

            if(dist_from_origin > max_length || dist_from_origin > len )
            {
                done = true;
                break;
            }
        }
    }

}

}
#endif // RAY_CAST_H
