
#ifndef SRC_VLP16_HELPER_H
#define SRC_VLP16_HELPER_H

#include "cuda_toolkit/cuda_macro.h"
#include "cuda_toolkit/projection.h"
#include "cuda_toolkit/occupancy/vlp16/multiscan_param.h"

namespace VLP_HELPER
{
    __device__ __forceinline__
    int positive_modulo(int i, int n)
    {
        return (i % n + n) % n;
    }

    //---
    __device__ __forceinline__
    float getDist2Line(float phi, float theta, float3 x0)
    {
        float3 unit_vector;
        unit_vector.z =sin(phi);
        unit_vector.x = cos(phi)*cos(theta);
        unit_vector.y = cos(phi)*sin(theta);

        float numerator = sqrtf((unit_vector.z*x0.y- unit_vector.y*x0.z)*(unit_vector.z*x0.y- unit_vector.y*x0.z)+
                               (unit_vector.x*x0.z- unit_vector.z*x0.x)*(unit_vector.x*x0.z- unit_vector.z*x0.x)+
                               (unit_vector.y*x0.x- unit_vector.x*x0.y)*(unit_vector.y*x0.x- unit_vector.x*x0.y));
        // denominator =1

        return numerator;
    }

    __device__ __forceinline__
    void G2L(const float3 &pos,const Projection &proj,
             const MulScanParam &param, const float &grid_width, int &theta_idx, int &phi_idx, LASER_RANGE_TPYE &depth)
    {
        // from global frame to sensor frame
        float3 local_pos = proj.G2L*pos;

        // calculate theta and theta_idx
        float theta = atan2f(local_pos.y,local_pos.x);
        theta_idx = floorf((theta - param.theta_min)/param.theta_inc + 0.5f);
        theta_idx = positive_modulo(theta_idx, param.scan_num);

        // calculate phi and phi_idx
        float range_hor = sqrtf(local_pos.y*local_pos.y + local_pos.x*local_pos.x);
        float phi = atan2(local_pos.z, range_hor);
        phi_idx = floor((phi - param.phi_min)/param.phi_inc +0.5f);

        // get the depth
        if(phi_idx<0 || phi_idx>= param.ring_num)
        {
            depth = -1.f;
            return;
        }
        float dist2ray= getDist2Line(phi, theta, local_pos);
        if(fabs(dist2ray)>=grid_width)
        {
            depth = -1.f;
            return;
        }

        depth=sqrtf(local_pos.x*local_pos.x+local_pos.y*local_pos.y);
    }

}

#endif //SRC_VLP16_HELPER_H
