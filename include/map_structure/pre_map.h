//
// Created by joseph on 23-2-16.
//

#ifndef SRC_PRE_MAP_H
#define SRC_PRE_MAP_H

#include <vector>
//#include "cuda_toolkit/occupancy/occupancy_map.h"
#include "cuda_toolkit/projection.h"
#include "voxmap_utils.cuh"
class Ext_Obs_Wrapper
{
public:
    std::vector<float3> rt_obsbbx_ll, rt_obsbbx_ur;
    thrust::device_vector<float3> obsbbx_ll_D;
    thrust::device_vector<float3> obsbbx_ur_D;
    thrust::device_vector<bool> obs_activated;
    int ext_obs_num;
public:
    Ext_Obs_Wrapper(int obs_num);
    void activate_AABB(float3& loc_map_ll, float3& loc_map_ur);
    void change_obs_num(int obs_num);
    bool CheckAABBIntersection(float3& A_ll, float3& A_ur, float3& B_ll, float3& B_ur);

    void assign_obs_premap(std::vector<float3>& preobs_ll, std::vector<float3>& preobs_ur);
    void append_new_elem(float3& ll_coord, float3& ur_coord);
    void bbx_H2D();
};

void warmupCuda();
#endif //SRC_PRE_MAP_H
