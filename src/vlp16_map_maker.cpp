
#include <sensor_msgs/point_cloud2_iterator.h>
#include "cuda_toolkit/occupancy/vlp16/vlp16_map_maker.h"
#include "kernel/vlp16/vlp16_interface.h"
//#define  USE_RS_LIDAR
Vlp16MapMaker::Vlp16MapMaker()
{


}

Vlp16MapMaker::~Vlp16MapMaker() {
    if(_initialized)
    {
        GPU_FREE(_gpu_mulscan);
    }
}

void Vlp16MapMaker::initialize(const MulScanParam &p)
{
    // Relase the old buffer if there is one
    if(_initialized)
        GPU_FREE(_gpu_mulscan);

    _mul_scan_param = p;
    size_t pynt_num=static_cast<size_t>(_mul_scan_param.scan_num)*static_cast<size_t>(_mul_scan_param.ring_num);
    _range_byte_sz = sizeof(LASER_RANGE_TPYE)*pynt_num;
    GPU_MALLOC(&_gpu_mulscan, _range_byte_sz);

    for(int i=0;i<_mul_scan_param.ring_num;i++)
    {
        scanlines[i].header.frame_id = "laser0";
        scanlines[i].angle_increment = _mul_scan_param.theta_inc;
        scanlines[i].angle_min = _mul_scan_param.theta_min;
        scanlines[i].angle_max = M_PI;
        scanlines[i].range_min = 0.0;
        scanlines[i].range_max = _mul_scan_param.max_r;
        scanlines[i].time_increment = 0.0;
        scanlines[i].ranges.resize(_mul_scan_param.scan_num, INFINITY);
        scanlines[i].intensities.resize(_mul_scan_param.scan_num);
    }

    _initialized = true;
}

void Vlp16MapMaker::setLocMap(LocMap *lMap) {

    _lMap = lMap;
}

void Vlp16MapMaker::updateLocalOGM(const Projection &proj, const sensor_msgs::PointCloud2ConstPtr& pyntcld,
                                   int3 *VB_keys_loc_D, const int time, bool for_motion_planner, int rbt_r2_grids) {
    // initialize 3d scans as far dist
    for(int j=0;j<_mul_scan_param.ring_num;j++)
    {
        std::fill(scanlines[j].ranges.begin(),scanlines[j].ranges.end(),INFINITY);
    }
    // convert point cloud to rings
    convertPyntCld(pyntcld);
    // copy 3d scan
    for(int j=0;j<_mul_scan_param.ring_num;j++)
    {
        int H_index =rayid_toup[j];
        LASER_RANGE_TPYE* H_scan_addr =(LASER_RANGE_TPYE*)(&scanlines[H_index].ranges.at(0));
        int data_offset = j*_mul_scan_param.scan_num;
        GPU_MEMCPY_H2D(&_gpu_mulscan[data_offset],H_scan_addr,_mul_scan_param.scan_num*sizeof(LASER_RANGE_TPYE));
    }
    // ogm kernel
    VLP_FAST::localOGMKernels(_lMap, _gpu_mulscan, proj, _mul_scan_param, VB_keys_loc_D,
                              for_motion_planner, rbt_r2_grids);
}

void Vlp16MapMaker::convertPyntCld(const sensor_msgs::PointCloud2ConstPtr &msg) {
    // Load structure of PointCloud2
    int offset_x = -1;
    int offset_y = -1;
    int offset_z = -1;
    int offset_i = -1;
    int offset_r = -1;

    for (size_t i = 0; i < msg->fields.size(); i++)
    {
        if (msg->fields[i].datatype == sensor_msgs::PointField::FLOAT32)
        {
            if (msg->fields[i].name == "x")
            {
                offset_x = msg->fields[i].offset;
            }
            else if (msg->fields[i].name == "y")
            {
                offset_y = msg->fields[i].offset;
            }
            else if (msg->fields[i].name == "z")
            {
                offset_z = msg->fields[i].offset;
            }
            else if (msg->fields[i].name == "intensity")
            {
                offset_i = msg->fields[i].offset;
            }
        }
        else if (msg->fields[i].datatype == sensor_msgs::PointField::UINT16)
        {
            if (msg->fields[i].name == "ring")
            {
                offset_r = msg->fields[i].offset;
            }
        }
    }
    // Construct LaserScan message
    if ((offset_x >= 0) && (offset_y >= 0) && (offset_r >= 0))
    {
        const float RESOLUTION = std::abs(_mul_scan_param.theta_inc);
        const size_t SIZE = _mul_scan_param.scan_num;

        if ((offset_x == 0) &&
            (offset_y == 4) &&
            (offset_i % 4 == 0) &&
            (offset_r % 4 == 0))
        {
            const size_t X = 0;
            const size_t Y = 1;
            const size_t Z = 2;
            const size_t I = offset_i / 4;
            const size_t R = offset_r / 4;
            for (sensor_msgs::PointCloud2ConstIterator<float> it(*msg, "x"); it != it.end(); ++it)
            {
                uint16_t r = *((const uint16_t*)(&it[R]));  // ring
                // NOTE: ring order can be  different on distinct lasers
#ifdef USE_RS_LIDAR
                if (r >= 8)
                    r = 23 - r;
#endif
                const float x = it[X];  // x
                const float y = it[Y];  // y
                const float z = it[Z];  // z
                const float i = it[I];  // intensity
                const int bin = (atan2f(y, x) + static_cast<float>(M_PI)) / RESOLUTION;

                if ((bin >= 0) && (bin < static_cast<int>(SIZE)))
                {
                    scanlines[r].ranges[bin] = sqrtf(x * x + y * y);
                    scanlines[r].intensities[bin] = i;
                }
            }
        }
    }
}