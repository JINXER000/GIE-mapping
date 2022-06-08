#ifndef HOKUYO_HELPER_H
#define HOKUYO_HELPER_H

#include "cuda_toolkit/cuda_macro.h"
#include "cuda_toolkit/projection.h"
#include "cuda_toolkit/occupancy/hokuyo/scan_param.h"

namespace SCAN_HELPER
{
__device__ __forceinline__
int positive_modulo(int i, int n)
{
  return (i % n + n) % n;
}

__device__ __forceinline__
void G2L(const float3 &pos,const Projection &proj,
                    const ScanParam &param, const float &grid_width, int &theta_idx, float &depth)
{
    // from global frame to sensor frame
    float3 local_pos = proj.G2L*pos;

    // calculate theta and theta_idx
    float theta = atan2f(local_pos.y,local_pos.x);
    theta_idx = floorf((theta - param.theta_min)/param.theta_inc + 0.5f);
    theta_idx = positive_modulo(theta_idx, param.scan_num);

    // get the depth
    if (fabsf(local_pos.z) < grid_width)
        depth=sqrtf(local_pos.x*local_pos.x+local_pos.y*local_pos.y);
    else
        depth = -1.f;
}
}


#endif // HOKUYO_HELPER_H
