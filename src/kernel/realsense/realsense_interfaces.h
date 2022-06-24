#ifndef REALSENSE_INTERFACES_H
#define REALSENSE_INTERFACES_H

#include "cuda_toolkit/projection.h"
#include "cuda_toolkit/occupancy/realsense/camera_param.h"
#include "map_structure/local_batch.h"

namespace REALSENSE_FAST
{
void localOGMKernels(LocMap* loc_map, REALSENSE_DEPTH_TPYE *detph_data, Projection proj, CamParam param,
                     int3* VB_keys_loc_D, bool for_motion_planner, int rbt_r2_grids);

}

#endif // REALSENSE_INTERFACES_H
