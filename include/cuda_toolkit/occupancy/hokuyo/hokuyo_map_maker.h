#ifndef HOKUYO_MAP_MAKER
#define HOKUYO_MAP_MAKER

#include <cuda_toolkit/occupancy/hokuyo/scan_param.h>
#include <cuda_toolkit/projection.h>
#include <sensor_msgs/LaserScan.h>
#include "map_structure/local_batch.h"

class HokuyoMapMaker
{
public:
    HokuyoMapMaker();
    ~HokuyoMapMaker();
    void initialize(const ScanParam &p);
    void initialize(const sensor_msgs::LaserScan::ConstPtr& msg);

    bool is_initialized(){return _initialized;}

    void setLocMap(LocMap *lMap);
    void updateLocalOGM(const Projection& proj,const sensor_msgs::LaserScan::ConstPtr &scan,
                        int3* VB_keys_loc_D, const int time,  bool for_motion_planner, int rbt_r2_grids);

private:
    ScanParam _scan_param;
    int _scan_byte_sz;
    SCAN_DEPTH_TPYE *_gpu_scan;
    bool _initialized = false;

    LocMap * _lMap;
};

#endif // HOKUYO_MAP_MAKER
