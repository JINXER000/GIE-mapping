
#ifndef SRC_VLP16_MAP_MAKER_H
#define SRC_VLP16_MAP_MAKER_H

#include "multiscan_param.h"
#include <cuda_toolkit/projection.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include "map_structure/local_batch.h"

class Vlp16MapMaker
{
public:
    Vlp16MapMaker();
    ~Vlp16MapMaker();

    void initialize(const MulScanParam &p);
    bool is_initialized(){return _initialized;}

    void setLocMap(LocMap *lMap);
    void updateLocalOGM(const Projection& proj, const sensor_msgs::PointCloud2ConstPtr& pyntcld,
                        int3* VB_keys_loc_D, const int time,  bool for_motion_planner, int rbt_r2_grids);
    void convertPyntCld(const sensor_msgs::PointCloud2ConstPtr& msg);
private:
    MulScanParam _mul_scan_param;
    int _range_byte_sz;
    SCAN_DEPTH_TPYE *_gpu_mulscan;
    bool _initialized = false;
    sensor_msgs::LaserScan  scanlines[16];
    const int rayid_toup[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    LocMap * _lMap;
};

#endif //SRC_VLP16_MAP_MAKER_H
