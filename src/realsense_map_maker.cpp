#include "cuda_toolkit/occupancy/realsense/realsense_map_maker.h"
#include "kernel/realsense/realsense_interfaces.h"
RealsenseMapMaker::RealsenseMapMaker()
{

}

RealsenseMapMaker::~RealsenseMapMaker()
{
    if(_initialized)
        GPU_FREE(_gpu_dep_img);
}

void RealsenseMapMaker::initialize(const CamParam &p)
{
    // Relase the old buffer if there is one
    if(_initialized)
        GPU_FREE(_gpu_dep_img);

    _cam_param = p;
    // Create device memory
    _img_byte_sz = (_cam_param.rows*_cam_param.cols)*sizeof(REALSENSE_DEPTH_TPYE);
    GPU_MALLOC(&_gpu_dep_img, _img_byte_sz);

    _initialized = true;
}

void RealsenseMapMaker::initialize(const sensor_msgs::CameraInfo::ConstPtr& msg, bool valid_NaN)
{
    float cx = static_cast<float>(msg->K[2]);
    float cy = static_cast<float>(msg->K[5]);
    float fx = static_cast<float>(msg->K[0]);
    float fy = static_cast<float>(msg->K[4]);
    int rows = static_cast<int>(msg->height);
    int cols = static_cast<int>(msg->width);

    CamParam p(rows,cols,cx,cy,fx,fy,valid_NaN);
    initialize(p);
}


void RealsenseMapMaker::setLocMap(LocMap *lMap)
{
    _lMap = lMap;
}
void RealsenseMapMaker::updateLocalOGM(const Projection& proj,const sensor_msgs::Image::ConstPtr &dep_img,
                                       int3* VB_keys_loc_D, const int time,  bool for_motion_planner, int rbt_r2_grids)
{
    REALSENSE_DEPTH_TPYE* head = (REALSENSE_DEPTH_TPYE*)&dep_img->data[0];
    GPU_MEMCPY_H2D(_gpu_dep_img,head,_img_byte_sz);
    REALSENSE_FAST::localOGMKernels(_lMap, _gpu_dep_img, proj, _cam_param,VB_keys_loc_D, for_motion_planner, rbt_r2_grids);
}