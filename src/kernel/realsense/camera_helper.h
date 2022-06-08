#ifndef CAMERA_HELPER_H
#define CAMERA_HELPER_H

#include "cuda_toolkit/cuda_macro.h"
#include "cuda_toolkit/projection.h"
#include "cuda_toolkit/occupancy/realsense/camera_param.h"

namespace CAM_HELPER
{
__device__ __forceinline__
void G2L(const float3 &pos,const Projection &proj,
                    const CamParam &param, int2 &pix, float &depth)
{
  // from global frame to sensor frame
  float3 local_pos = proj.G2L*pos;

  // get the depth
  depth=local_pos.x;

  // calculate the pixel coordinate on the image
  pix.x=floorf(-local_pos.y*param.fx/depth+param.cx+0.5f);
  pix.y=floorf(-local_pos.z*param.fy/depth+param.cy+0.5f);
}

__device__ __forceinline__
void L2G(const int2 &pix, const float &depth, const Projection &proj, const CamParam &param,
                  float3 & pos)
{
  //From pix to camera frame
  float3 local_pos;
  local_pos.x = depth;
  local_pos.y = (param.cx - pix.x)*depth/param.fx;
  local_pos.z = (param.cy - pix.y)*depth/param.fy;

  pos = proj.L2G*local_pos;
}
}

#endif // CAMERA_HELPER_H
