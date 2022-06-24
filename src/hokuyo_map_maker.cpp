#include "cuda_toolkit/occupancy/hokuyo/hokuyo_map_maker.h"
#include "kernel/hokuyo/hokuyo_interfaces.h"
HokuyoMapMaker::HokuyoMapMaker()
{

}

HokuyoMapMaker::~HokuyoMapMaker()
{
    if(_initialized)
        GPU_FREE(_gpu_scan);
}

void HokuyoMapMaker::initialize(const ScanParam &p)
{
    // Relase the old buffer if there is one
    if(_initialized)
        GPU_FREE(_gpu_scan);

    _scan_param = p;
    // Create device memory
    _scan_byte_sz = (_scan_param.scan_num)*sizeof(SCAN_DEPTH_TPYE);
    GPU_MALLOC(&_gpu_scan, _scan_byte_sz);

    _initialized = true;
}

void HokuyoMapMaker::initialize(const sensor_msgs::LaserScan::ConstPtr& msg)
{
    ScanParam p;
    p.scan_num = msg->ranges.size();
    p.max_r = msg->range_max;
    p.theta_inc = msg->angle_increment;
    p.theta_min = msg->angle_min;

    initialize(p);
}

void HokuyoMapMaker::setLocMap(LocMap *lMap)
{
    _lMap = lMap;
}

void HokuyoMapMaker::updateLocalOGM(const Projection& proj,const sensor_msgs::LaserScan::ConstPtr &scan,
                                    int3* VB_keys_loc_D, const int time,  bool for_motion_planner, int rbt_r2_grids)
{
    SCAN_DEPTH_TPYE* head=(SCAN_DEPTH_TPYE*)(&scan->ranges.at(0));
    GPU_MEMCPY_H2D(_gpu_scan,head,_scan_byte_sz);
    HOKUYO_FAST::localOGMKernels(_lMap,_gpu_scan,proj,_scan_param,VB_keys_loc_D, for_motion_planner, rbt_r2_grids);
}
