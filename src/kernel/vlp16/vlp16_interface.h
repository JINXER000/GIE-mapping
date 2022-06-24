

#ifndef SRC_VLP16_INTERFACE_H
#define SRC_VLP16_INTERFACE_H

#include "cuda_toolkit/projection.h"
#include "cuda_toolkit/occupancy/vlp16/multiscan_param.h"
#include "map_structure/local_batch.h"

namespace VLP_FAST
{
    void localOGMKernels(LocMap* loc_map, SCAN_DEPTH_TPYE *detph_data, Projection proj, MulScanParam param,
                         int3* VB_keys_loc_D, bool for_motion_planner, int rbt_r2_grids);

}
#endif //SRC_VLP16_INTERFACE_H
