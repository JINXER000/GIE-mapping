#ifndef HOKUYO_INTERFACES_H
#define HOKUYO_INTERFACES_H

#include "cuda_toolkit/projection.h"
#include "cuda_toolkit/occupancy/hokuyo/scan_param.h"
#include "map_structure/local_batch.h"

namespace HOKUYO_FAST
{
void localOGMKernels(LocMap* loc_map, SCAN_DEPTH_TPYE *detph_data, Projection proj, ScanParam param,
                     int3* VB_keys_loc_D, bool for_motion_planner, int rbt_r2_grids);

}

#endif // HOKUYO_INTERFACES_H
