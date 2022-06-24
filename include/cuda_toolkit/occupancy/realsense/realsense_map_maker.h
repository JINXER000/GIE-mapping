#ifndef REALSENSE_MAP_MAKER_H
#define REALSENSE_MAP_MAKER_H

#include <cuda_toolkit/occupancy/realsense/camera_param.h>
#include <cuda_toolkit/projection.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include "map_structure/local_batch.h"

class RealsenseMapMaker
{
public:
    RealsenseMapMaker();
    ~RealsenseMapMaker();
    void initialize(const CamParam &p);
    void initialize(const sensor_msgs::CameraInfo::ConstPtr& msg, bool valid_NaN);
    void setLocMap(LocMap *lMap);
    void updateLocalOGM(const Projection& proj,const sensor_msgs::Image::ConstPtr &dep_img,
                        int3* VB_keys_loc_D, const int time, bool for_motion_planner, int rbt_r2_grids);
    bool is_initialized(){return _initialized;}
private:
    CamParam _cam_param;
    LocMap *_lMap;
    int _img_byte_sz;
    REALSENSE_DEPTH_TPYE *_gpu_dep_img;
    bool _initialized = false;
};

#endif // REALSENSE_MAP_MAKER_H
