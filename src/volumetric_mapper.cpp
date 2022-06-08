#include "volumetric_mapper.h"
VOLMAPNODE::VOLMAPNODE()
{
    // Load parameters
    param.setupParam(_nh);

    // Publisher
    edt_msg_pub = _nh.advertise<GIE::CostMap> ("cost_map", 1);

    // Subscriber
    if(param.data_case == "ugv_corridor" || param.data_case == "laser3D")
    {
        s_odom_sub.subscribe(_nh,"/LaserOdomTopic", 1);
        s_pntcld_sub.subscribe(_nh,"/rslidar_points", 1);
        pntcld_sync = new message_filters::Synchronizer<pntcld_sync_policy> (pntcld_sync_policy(30), s_pntcld_sub, s_odom_sub);
        pntcld_sync->setMaxIntervalDuration(ros::Duration(0.1));
        pntcld_sync->registerCallback(boost::bind(&VOLMAPNODE::CB_pntcld_odom,this,_1,_2));
    }else if(param.data_case == "cow_lady")
    {
        s_trfm_sub.subscribe(_nh,"/kinect/vrpn_client/estimated_transform", 1);
        s_pntcld_sub.subscribe(_nh,"/camera/depth_registered/points", 1);
        cow_sync = new message_filters::Synchronizer<cow_sync_policy> (cow_sync_policy (30), s_pntcld_sub, s_trfm_sub);
        cow_sync->setMaxIntervalDuration(ros::Duration(0.1));
        cow_sync->registerCallback(boost::bind(&VOLMAPNODE::CB_cow,this,_1,_2));
    }else if(param.data_case == "depth_cam")
    {
        _caminfo_sub = _nh.subscribe("/iris_0/kinect_ir/depth/camera_info",1,&VOLMAPNODE::CB_caminfo, this);
        // sync sub
        s_odom_sub.subscribe(_nh,"/iris_0/mavros/local_position/odom", 1);
        s_depth_sub.subscribe(_nh,"/iris_0/kinect_ir/kinect/depth/image_raw", 1);
        depth_sync = new message_filters::Synchronizer<depth_sync_policy> (depth_sync_policy(30), s_depth_sub, s_odom_sub);
        depth_sync->setMaxIntervalDuration(ros::Duration(0.1));
        depth_sync->registerCallback(boost::bind(&VOLMAPNODE::CB_depth_odom,this,_1,_2));
    }else if(param.data_case == "laser2D")
    {
        // sync sub
        s_odom_sub.subscribe(_nh,"/odom", 1);
        s_laser_sub.subscribe(_nh,"/scan", 1);
        laser2D_sync = new message_filters::Synchronizer<laser2D_sync_policy> (laser2D_sync_policy(30), s_laser_sub, s_odom_sub);
        laser2D_sync->setMaxIntervalDuration(ros::Duration(0.1));
        laser2D_sync->registerCallback(boost::bind(&VOLMAPNODE::CB_scan_odom,this,_1,_2));
    }


    // Normal initialization
    _laser_ptr = boost::shared_ptr<sensor_msgs::LaserScan>(new sensor_msgs::LaserScan());
    _odom_ptr = boost::shared_ptr<nav_msgs::Odometry>(new nav_msgs::Odometry());
    _depth_ptr = boost::shared_ptr<sensor_msgs::Image>(new sensor_msgs::Image());
    _pntcld_ptr = boost::shared_ptr<sensor_msgs::PointCloud2>(new sensor_msgs::PointCloud2());
    _time = 0;

    int3 local_grids = make_int3(param.local_size_x/param.voxel_width, param.local_size_y/param.voxel_width, param.local_size_z/param.voxel_width);


    // setup local map
    _loc_map = new LocMap(param.voxel_width, local_grids, param.occupancy_threshold, param.ogm_min_h, param.ogm_max_h, param.cutoff_grids_sq);
    _pnt_map_maker.setLocMap(_loc_map);
    _rea_map_maker.setLocMap(_loc_map);
    _hok_map_maker.setLocMap(_loc_map);
    _loc_map->create_gpu_map();
    setupRotationPlan();
    
    // Setup glb EDT map
    _hash_map = new GlbHashMap(_loc_map->_bdr_num,_loc_map->_update_size, param.max_bucket, param.max_block);
    _hash_map->setLocMap(_loc_map);

    // for display
    if (param.display_occ)
    {
        _occ_rviz_pub = _nh.advertise<PntCldI>("occ_map",1);
        _occ_pnt_cld = PntCldI::Ptr(new PntCldI);
        _occ_pnt_cld->header.frame_id = "/map";
    }

    if (param.display_loc_edt)
    {
        _edt_rviz_pub = _nh.advertise<PntCldI>("edt_map",1);
        _edt_pnt_cld = PntCldI::Ptr(new PntCldI);
        _edt_pnt_cld->header.frame_id = "/map";
    }

    if(param.display_glb_edt)
    {
        _glb_edt_rviz_pub = _nh.advertise<PntCldI>("glb_edt_map", 1);
        _glb_edt_pnt_cld = PntCldI ::Ptr(new PntCldI);
        _glb_edt_pnt_cld->header.frame_id = "/map";
    }

    if(param.display_glb_ogm)
    {
        _glb_ogm_rviz_pub = _nh.advertise<PntCld>("glb_ogm_map", 1);
        _glb_ogm_pnt_cld = PntCld ::Ptr(new PntCld);
        _glb_ogm_pnt_cld->header.frame_id = "/map";
    }


    if(param.for_motion_planner)
        setupEDTmsg4Motion(cost_map_msg, _loc_map, true);

    // logger
    logger = new csvfile(param.log_dir);
    (*logger)<<"Occupancy time"<<"EDT time";


    // Timer
    _mapTimer = _nh.createTimer(ros::Duration(0.5), &VOLMAPNODE::publishMap, this);

}
//---
VOLMAPNODE::~VOLMAPNODE()
{

}
//---
void VOLMAPNODE::publishMap(const ros::TimerEvent&)
{

    if(!_msg_mgr.is_ready())
        return;

    _time++;


    tf::Transform trans = odom2trans();
    Projection proj = trans2proj(trans);

    if (param.data_case == "ugv_corridor")
    {
        proj.origin.z = 1;
    }

    if(mtx.try_lock() == false)
    {
        printf("!!!Please decrease frequency!!!\n");
        assert(false);
    }

    // construct OGM
    auto start = std::chrono::steady_clock::now();
    _loc_map->calculate_pivot_origin(proj.origin);
    if (_msg_mgr.got_dep_img)
    {
        _rea_map_maker.updateLocalOGM(proj, _depth_ptr, thrust::raw_pointer_cast(_hash_map->VB_keys_loc_D.data()), _time,
                                      param.for_motion_planner, param.robot_r2_grids);
    }
    else if(_msg_mgr.got_scan)
    {
        _hok_map_maker.updateLocalOGM(proj, _laser_ptr, thrust::raw_pointer_cast(_hash_map->VB_keys_loc_D.data()), _time,
                                      param.for_motion_planner, param.robot_r2_grids);
    }
    else if(_msg_mgr.got_pnt_cld)
    {
        _pnt_map_maker.updateLocalOGM(proj,_pntcld_ptr, thrust::raw_pointer_cast(_hash_map->VB_keys_loc_D.data()), _time);
    }

    _hash_map->updateHashOGM(_msg_mgr.got_pnt_cld, _time,param.display_glb_ogm && !param.display_glb_edt);


    GPU_DEV_SYNC(); // only for profiling, can be deleted for faster speed
    auto end= std::chrono::steady_clock::now();
    auto ogm_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    (*logger)<<endrow<<(float)ogm_duration;

    // construct global EDT 
    start = std::chrono::steady_clock::now();

    EDT_OCC::batchEDTUpdate(_loc_map, _rotation_plan, _time);

    _hash_map->mergeNewObsv( _time, param.display_glb_edt);
    if(param.display_glb_edt || param.display_glb_ogm)
        _hash_map->streamPipeline();

    end = std::chrono::steady_clock::now();
    auto edt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    (*logger)<<(float)edt_duration;
    std::cout << "OGM time: "<< ogm_duration<< "ms; Global EDT time: "<< edt_duration<< " ms" << std::endl;

    mtx.unlock();

    // publish cost map msg
    if(param.for_motion_planner)
    {
        _loc_map->convertCostMap();
        memcpy(cost_map_msg.payload8.data(),_loc_map->seendist_out,static_cast<size_t>(_loc_map->seendist_size));
        setupEDTmsg4Motion(cost_map_msg, _loc_map, false);
        edt_msg_pub.publish (cost_map_msg);
    }


    //do visualization
    auto vis_start = std::chrono::steady_clock::now();
    static int vis_cnt = 0;
    if(vis_cnt%param.vis_interval ==0)
        visualize(proj.origin);
    vis_cnt++;
    auto vis_end= std::chrono::steady_clock::now();
    auto vis_duration = std::chrono::duration_cast<std::chrono::milliseconds>(vis_end - vis_start).count();
    std::cout << "Visualize & profile time: "<< vis_duration<< "ms"<< std::endl;
}


tf::Transform VOLMAPNODE::odom2trans()  {
    // Construct a transfrom from body to world using current the pose
    tf::Quaternion quat;
    tf::quaternionMsgToTF(_odom_ptr->pose.pose.orientation, quat);
    tf::Transform trans;
    trans.setRotation(quat);
    trans.setOrigin(tf::Vector3(_odom_ptr->pose.pose.position.x,
                                _odom_ptr->pose.pose.position.y,
                                _odom_ptr->pose.pose.position.z));

    if (param.data_case == "cow_lady")
    {
        // transform correction

        Eigen::Affine3d transEigen;
        tf::transformTFToEigen( trans, transEigen);
        Eigen::Matrix4d transMat = transEigen.matrix();
        transMat = transMat*param.T_V_C;
        transEigen.matrix() = transMat;
        tf::Transform transCorrect;
        tf::transformEigenToTF(transEigen, transCorrect);
        return  transCorrect;
    }
    return trans;
}

void VOLMAPNODE::CB_pntcld_odom(const sensor_msgs::PointCloud2::ConstPtr &pcld, const nav_msgs::Odometry::ConstPtr &odom)
{
    CB_odom(odom);
    CB_pntcld(pcld);
}

void VOLMAPNODE::CB_cow(const sensor_msgs::PointCloud2::ConstPtr &pcld, const geometry_msgs::TransformStamped::ConstPtr &trfm)
{

    // transform to odom
    _odom_ptr->header.frame_id = "world";
    _odom_ptr->header.stamp=trfm->header.stamp;
    _odom_ptr->pose.pose.position.x =trfm->transform.translation.x;
    _odom_ptr->pose.pose.position.y = trfm->transform.translation.y;
    _odom_ptr->pose.pose.position.z = trfm->transform.translation.z;
    _odom_ptr->pose.pose.orientation = trfm->transform.rotation;
    _msg_mgr.got_odom = true;

    CB_pntcld(pcld);
}

void VOLMAPNODE::CB_depth_odom(const sensor_msgs::Image::ConstPtr &img, const nav_msgs::Odometry::ConstPtr &odom)
{
    CB_odom(odom);
    CB_depth(img);
}

void VOLMAPNODE::CB_scan_odom(const sensor_msgs::LaserScan::ConstPtr &scan, const nav_msgs::Odometry::ConstPtr &odom)
{
    CB_odom(odom);
    CB_scan2D(scan);
}

//---
void VOLMAPNODE::CB_caminfo(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
    if(!_rea_map_maker.is_initialized())
        _rea_map_maker.initialize(msg, true);
}
//---
void VOLMAPNODE::CB_odom(const nav_msgs::Odometry::ConstPtr &msg)
{
    *_odom_ptr = *msg;
    _msg_mgr.got_odom = true;
}
//---
void VOLMAPNODE::CB_depth(const sensor_msgs::Image::ConstPtr& msg)
{
    if(_rea_map_maker.is_initialized())
    {
        *_depth_ptr = *msg;
        _msg_mgr.got_dep_img = true;
    }
}
//---
void VOLMAPNODE::CB_scan2D(const sensor_msgs::LaserScan::ConstPtr& msg)
{
    if(!_hok_map_maker.is_initialized())
    {
        _hok_map_maker.initialize(msg);
    }

    *_laser_ptr = *msg;
    _msg_mgr.got_scan = true;

}
//---
void VOLMAPNODE::CB_pntcld(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    if(!_pnt_map_maker.is_initialized())
    {
        _pnt_map_maker.initialize(msg);
    }

    *_pntcld_ptr = *msg;
    _msg_mgr.got_pnt_cld = true;

}
//---
void VOLMAPNODE::setupRotationPlan()
{
    // Setup the cutt rotations
    if (_loc_map->_update_size.z == 1)
    {
        int Dx = _loc_map->_update_size.x;
        int Dy = _loc_map->_update_size.y;
        int dimension_0[2] = {Dx,Dy};
        int permutation_0[2] = {1,0};
        int dimension_1[2] = {Dy,Dx};
        int permutation_1[2] = {1,0};
        cuttPlan(&_rotation_plan[0], 2, dimension_0, permutation_0, sizeof(int), nullptr);
        cuttPlan(&_rotation_plan[1], 2, dimension_1, permutation_1, sizeof(int), nullptr);
    }
    else
    {
        int Dx = _loc_map->_update_size.x;
        int Dy = _loc_map->_update_size.y;
        int Dz = _loc_map->_update_size.z;
        int dimension_0[3] = {Dx,Dy,Dz};
        int permutation_0[3] = {1,0,2};
        int dimension_1[3] = {Dy,Dx,Dz};
        int permutation_1[3] = {0,2,1};
        int dimension_2[3] = {Dy,Dz,Dx};
        int permutation_2[3] = {2,0,1};
        cuttPlan(&_rotation_plan[0], 3, dimension_0, permutation_0, sizeof(int), nullptr);
        cuttPlan(&_rotation_plan[1], 3, dimension_1, permutation_1, sizeof(int), nullptr);
        cuttPlan(&_rotation_plan[2], 3, dimension_2, permutation_2, sizeof(int), nullptr);
    }
}

void VOLMAPNODE::setupEDTmsg4Motion(GIE::CostMap &msg, LocMap* loc_map, bool resize) {
    msg.x_origin = loc_map->_msg_origin.x;
    msg.y_origin = loc_map->_msg_origin.y;
    msg.z_origin = loc_map->_msg_origin.z;
    msg.width = loc_map->_voxel_width;

    if(resize)
    {
        msg.x_size = loc_map->_update_size.x;
        msg.y_size = loc_map->_update_size.y;
        msg.z_size = loc_map->_update_size.z;
        msg.payload8.resize(sizeof(SeenDist)*static_cast<unsigned int>(msg.x_size*msg.y_size*msg.z_size));
    }
    msg.type = GIE::CostMap::TYPE_EDT;
}

