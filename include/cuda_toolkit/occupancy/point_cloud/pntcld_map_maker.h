#ifndef PNTCLD_MAP_MAKER_H
#define PNTCLD_MAP_MAKER_H

#include <cuda_toolkit/occupancy/point_cloud/pntcld_param.h>
#include <cuda_toolkit/projection.h>
#include <sensor_msgs/PointCloud2.h>
#include "map_structure/local_batch.h"

class PntcldMapMaker
{
public:
    PntcldMapMaker();
    ~PntcldMapMaker();
    void initialize(const PntcldParam &p);
    void initialize(const sensor_msgs::PointCloud2::ConstPtr& msg);
    void setLocMap(LocMap *lMap);
    void updateLocalOGM(const Projection& proj,const sensor_msgs::PointCloud2::ConstPtr &msg,
                        int3* VB_keys_loc_D, const int time, bool for_motion_planner, int rbt_r2_grids);
    bool is_initialized(){return _initialized;}

    void pntcld_process(const sensor_msgs::PointCloud2ConstPtr& msg);

private:
    PntcldParam _pnt_param;
    LocMap *_lMap;
    int _cld_byte_sz;
    PNT_TYPE *_gpu_cld;
    PNT_TYPE *_cpu_cld;
    bool _initialized = false;
};


#endif // PNTCLD_MAP_MAKER_H
