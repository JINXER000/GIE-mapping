#ifndef SRC_PARAMETERS_H
#define SRC_PARAMETERS_H
#include <ros/ros.h>
// user defines for developer

/*!
 * parameters that can be modified in yaml
 */
struct Parameters
{
    bool for_motion_planner;
    float robot_r;
    int robot_r2_grids;

    std::string  log_dir = "/home/joseph/GIE_log.csv";
    // incoming data case
    std::string data_case = "laser2D";


    //    display option
    bool display_loc_edt ;
    bool display_occ ;
    bool display_glb_edt;
    bool display_glb_ogm;
    // do visualization every n map tick
    int vis_interval;
    // 0~255, considered as obstacle if val is greater
    int occupancy_threshold;

    float vis_height;
    // voxel width (m)
    float voxel_width;
    // local volume size (m)
    float local_size_x, local_size_y, local_size_z;
    float ogm_min_h, ogm_max_h;
    // stop propagate if dist is greater
    float cutoff_dist;
    int cutoff_grids_sq;

    // num of buckets in GPU-hashtable
    int max_bucket;
    // num of blocks in GPU-hashtable
    int max_block;

    // for cow-lady: T_V_C (C = cam0, V = vicon marker)
    Eigen::Matrix4d T_V_C;
    /**
     * Load all parameters
     * @param nh
     */
    void setupParam(const ros::NodeHandle &nh)
    {
        // General
        nh.param<bool>("GIE_mapping/for_motion_planner",for_motion_planner,true);
        nh.param<bool>("GIE_mapping/display_glb_edt",display_glb_edt,true);
        nh.param<bool>("GIE_mapping/display_glb_ogm",display_glb_ogm,true);
        nh.param<bool>("GIE_mapping/display_loc_edt",display_loc_edt,true);
        nh.param<bool>("GIE_mapping/display_occ",display_occ,false);
        nh.param<int>("GIE_mapping/vis_interval",vis_interval,1);
        nh.param<int>("GIE_mapping/occupancy_threshold",occupancy_threshold,180);
        nh.param<float>("GIE_mapping/vis_height",vis_height,1);
        nh.param<float>("GIE_mapping/voxel_width",voxel_width,0.1);
        nh.param<float>("GIE_mapping/local_size_x",local_size_x,10);
        nh.param<float>("GIE_mapping/local_size_y",local_size_y,10);
        nh.param<float>("GIE_mapping/local_size_z",local_size_z,3);

        nh.param<float>("GIE_mapping/ogm/min_height",ogm_min_h,-10);
        nh.param<float>("GIE_mapping/ogm/max_height",ogm_max_h,10);
        // wavefront
        nh.param<float>("GIE_mapping/wave/cutoff_dist",cutoff_dist,2);
        cutoff_grids_sq = flt2GridsSq(cutoff_dist);

        nh.param<float>("GIE_mapping/robot_r",robot_r,0.2);
        robot_r2_grids = flt2GridsSq(robot_r);

        // voxhashing
        nh.param<int>("GIE_mapping/hash/max_bucket",max_bucket,10000);
        nh.param<int>("GIE_mapping/hash/max_block",max_block,19997);

        // logger
        nh.getParam("GIE_mapping/log_dir", log_dir);
        // dataset
        nh.getParam("/data_case", data_case);
        std::cout<<"data_case is "<<data_case<<std::endl;
        std::cout<<"voxel_width is "<<voxel_width<<std::endl;
        if (data_case == "cow_lady")
        {
            T_V_C << 0.971048, -0.120915, 0.206023, 0.00114049,
                    0.15701, 0.973037, -0.168959, 0.0450936,
                    -0.180038, 0.196415, 0.96385, 0.0430765,
                    0.0, 0.0, 0.0, 1.0;
        }
    }

    int flt2GridsSq(float rad)
    {
        int grds = std::ceil(rad/voxel_width);
        return  grds*grds;
    }
};
#endif //SRC_PARAMETERS_H
