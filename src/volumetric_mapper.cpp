

#include "volumetric_mapper.h"
#include "warmup.h"

VOLMAPNODE::VOLMAPNODE()
{
    // Load parameters
    param.setupParam(_nh);

    // Publisher
    edt_msg_pub = _nh.advertise<GIE::CostMap> ("cost_map", 1);

    ext_obs = new Ext_Obs_Wrapper(param.obsbbx_ur.size());
    ext_obs->assign_obs_premap(param.obsbbx_ll, param.obsbbx_ur);
//    ext_obs->bbx_H2D();

    // Subscriber
    if(param.data_case == "ugv_corridor" )
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
    }else if(param.data_case == "laser3D")
    {
        s_odom_sub.subscribe(_nh,"/odom", 1);
        s_pntcld_sub.subscribe(_nh,"/velodyne_points", 1);
        pntcld_sync = new message_filters::Synchronizer<pntcld_sync_policy> (pntcld_sync_policy(30), s_pntcld_sub, s_odom_sub);
        pntcld_sync->setMaxIntervalDuration(ros::Duration(0.1));
        pntcld_sync->registerCallback(boost::bind(&VOLMAPNODE::CB_pntcld_odom,this,_1,_2));
    }

    // receive external knowledge form pyntcld
    ext_cld_sub = _nh.subscribe("/forbid_reg_cloud", 1, &VOLMAPNODE::CB_ext_cld, this);


    // Normal initialization
    _laser_ptr = boost::shared_ptr<sensor_msgs::LaserScan>(new sensor_msgs::LaserScan());
    _odom_ptr = boost::shared_ptr<nav_msgs::Odometry>(new nav_msgs::Odometry());
    _depth_ptr = boost::shared_ptr<sensor_msgs::Image>(new sensor_msgs::Image());
    _pntcld_ptr = boost::shared_ptr<sensor_msgs::PointCloud2>(new sensor_msgs::PointCloud2());
    _time = 0;

    int3 local_grids = make_int3(param.local_size_x/param.voxel_width, param.local_size_y/param.voxel_width, param.local_size_z/param.voxel_width);

    // setup local map
    _loc_map = new LocMap(param.voxel_width, local_grids, param.occupancy_threshold, param.ogm_min_h, param.ogm_max_h,
                          param.cutoff_grids_sq, param.fast_mode);
    _pnt_map_maker.setLocMap(_loc_map);
    _rea_map_maker.setLocMap(_loc_map);
    _hok_map_maker.setLocMap(_loc_map);
    _vlp_map_maker.setLocMap(_loc_map);
    _loc_map->create_gpu_map();
    setupRotationPlan();
    // Setup glb EDT map
    _hash_map = new GlbHashMap(_loc_map->_bdr_num,_loc_map->_local_size, param.max_bucket, param.max_block);
    _hash_map->setLocMap(_loc_map);

    if (param.display_loc_ogm)
    {
        _occ_rviz_pub = _nh.advertise<PntCldI>("occ_map",1);
        _occ_pnt_cld = PntCldI::Ptr(new PntCldI);
        _occ_pnt_cld->header.frame_id = "map";
    }

    if (param.display_loc_edt)
    {
        _edt_rviz_pub = _nh.advertise<PntCldI>("edt_map",1);
        _edt_pnt_cld = PntCldI::Ptr(new PntCldI);
        _edt_pnt_cld->header.frame_id = "map";
    }

    if(param.display_glb_edt)
    {
        _glb_edt_rviz_pub = _nh.advertise<PntCldI>("glb_edt_map", 1);
        _glb_edt_pnt_cld = PntCldI ::Ptr(new PntCldI);
        _glb_edt_pnt_cld->header.frame_id = "map";
    }

    if(param.display_glb_ogm)
    {
        _glb_ogm_rviz_pub = _nh.advertise<PntCld>("glb_ogm_map", 1);
        _glb_ogm_pnt_cld = PntCld ::Ptr(new PntCld);
        _glb_ogm_pnt_cld->header.frame_id = "map";
    }

    _dbg_rviz_pub =  _nh.advertise<PntCldI>("dbg_pt", 1);
    _dbg_pnt_cld = PntCldI ::Ptr(new PntCldI);
    _dbg_pnt_cld->header.frame_id = "map";

    if(param.for_motion_planner)
        setupEDTmsg4Motion(cost_map_msg, _loc_map, true);

    // profile
    logger = new csvfile(param.log_dir);
    (*logger)<<"Occupancy time"<<"EDT time"<< "RMSE";

    gtc = new Gnd_truth_checker();

    warmupCuda();

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

    if (param.ugv_height >0)
    {
        proj.origin.z = param.ugv_height;
    }

    auto start = std::chrono::steady_clock::now();
    _loc_map->calculate_pivot_origin(proj.origin);
    _loc_map->calculate_update_pivot(proj.origin);
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
        if(param.data_case == "laser3D")
        {
            _vlp_map_maker.updateLocalOGM(proj,_pntcld_ptr, thrust::raw_pointer_cast(_hash_map->VB_keys_loc_D.data()), _time,
                                          param.for_motion_planner, param.robot_r2_grids);
        }else
        {
            _pnt_map_maker.updateLocalOGM(proj,_pntcld_ptr, thrust::raw_pointer_cast(_hash_map->VB_keys_loc_D.data()), _time,
                                          param.for_motion_planner, param.robot_r2_grids);
        }
    }

    update_ext_map();

    _hash_map->updateHashOGM(_msg_mgr.got_pnt_cld  && param.data_case != "laser3D",
                             _time,param.display_glb_ogm && !param.display_glb_edt,
                             ext_obs);


    GPU_DEV_SYNC(); // only for profiling
    auto end= std::chrono::steady_clock::now();
    auto ogm_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    (*logger)<<endrow<<(float)ogm_duration;

    start = std::chrono::steady_clock::now();


    EDT_OCC::batchEDTUpdate(_loc_map, _rotation_plan, _time);

    _hash_map->mergeNewObsv( _time, param.display_glb_edt);
    if(param.display_glb_edt || param.display_glb_ogm)
        _hash_map->streamPipeline();

    end = std::chrono::steady_clock::now();
    auto edt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    (*logger)<<(float)edt_duration;
    std::cout << "OGM time: "<< ogm_duration<< "ms; Global EDT time: "<< edt_duration<< " ms" << std::endl;


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

    br.sendTransform(tf::StampedTransform(trans, cur_stamp, "map", "laser_3d"));
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
    if(param.data_case=="laser3D")
    {
        if(!_vlp_map_maker.is_initialized())
        {
            MulScanParam mp(440,16,10.0f, 2.0 * M_PI/440, -M_PI, 2.0f/180.0f *M_PI, -15.0f/180.0f*M_PI);
            _vlp_map_maker.initialize(mp);
        }
    }else
    {
        if(!_pnt_map_maker.is_initialized())
        {
            _pnt_map_maker.initialize(msg);
        }
    }

    cur_stamp = msg->header.stamp;
    *_pntcld_ptr = *msg;
    _msg_mgr.got_pnt_cld = true;

}
//---
void VOLMAPNODE::setupRotationPlan()
{
    // Setup the cutt rotations
    if (_loc_map->_local_size.z == 1)
    {
        int Dx = _loc_map->_local_size.x;
        int Dy = _loc_map->_local_size.y;
        int dimension_0[2] = {Dx,Dy};
        int permutation_0[2] = {1,0};
        int dimension_1[2] = {Dy,Dx};
        int permutation_1[2] = {1,0};
        cuttPlan(&_rotation_plan[0], 2, dimension_0, permutation_0, sizeof(int), nullptr);
        cuttPlan(&_rotation_plan[1], 2, dimension_1, permutation_1, sizeof(int), nullptr);
    }
    else
    {
        int Dx = _loc_map->_local_size.x;
        int Dy = _loc_map->_local_size.y;
        int Dz = _loc_map->_local_size.z;
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
        msg.x_size = loc_map->_local_size.x;
        msg.y_size = loc_map->_local_size.y;
        msg.z_size = loc_map->_local_size.z;
        msg.payload8.resize(sizeof(SeenDist)*static_cast<unsigned int>(msg.x_size*msg.y_size*msg.z_size));
    }
    msg.type = GIE::CostMap::TYPE_EDT;
}

void VOLMAPNODE::clustring(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    // TODO: below is to reset the obs to premap
    ext_obs->assign_obs_premap(param.obsbbx_ll, param.obsbbx_ur);

    pcl::fromROSMsg(*msg, ext_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < ext_cloud.size(); i++)
    {
        input_cloud->push_back(ext_cloud.points[i]);
    }
    input_cloud->header = ext_cloud.header;

    // construct KD tree
    if (input_cloud->points.size() != 0) {

        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        kdtree->setInputCloud(input_cloud);

        /* DBSCAN param*/
        int min_nbrPts = 3;
        double search_rad = 0.3;
        std::vector<pcl::PointIndices> clustered_indices;
        std::vector<int> pts_type(input_cloud->points.size(), 0);    // not proc-0；under proc-1；finished-2；
        std::vector<int> nbrs_idx;
        std::vector<float> nbrs_distance;


        for (int i = 0; i < input_cloud->points.size(); i++) {
            if (pts_type[i] == 2) { continue; }

            int nbrs_size = kdtree->radiusSearch(i, search_rad, nbrs_idx, nbrs_distance);

            std::vector<int> seed_queue;
            seed_queue.push_back(i);
            pts_type[i] = 2;
            for (int j = 0; j < nbrs_size; j++) {
                if (nbrs_idx[j] != i)
                {
                    seed_queue.push_back(nbrs_idx[j]);
                    pts_type[nbrs_idx[j]] = 1;
                }
            }

            int sq_idx = 1;
            while (sq_idx < seed_queue.size()) {
                int pt_idx = seed_queue[sq_idx];
                if (pts_type[pt_idx] == 2)
                {
                    sq_idx++;
                    continue;
                }
                nbrs_size = kdtree->radiusSearch(pt_idx, search_rad, nbrs_idx, nbrs_distance);
                if (nbrs_size >= min_nbrPts)
                {
                    for (int j = 0; j < nbrs_size; j++) {
                        if (pts_type[nbrs_idx[j]] == 0) {
                            seed_queue.push_back(nbrs_idx[j]);
                            pts_type[nbrs_idx[j]] = 1;
                        }
                    }
                }
                pts_type[pt_idx] = 2;
                sq_idx++;
            }

            pcl::PointIndices pt_idc;
            if (seed_queue.size() >= 4) {
                pt_idc.indices.resize(seed_queue.size());
                for (int j = 0; j < seed_queue.size(); j++) {
                    pt_idc.indices[j] = seed_queue[j];
                }
                pt_idc.header = input_cloud->header;
                clustered_indices.push_back(pt_idc);
            }
        }

        int j = 0;
        for (std::vector<pcl::PointIndices>::const_iterator it = clustered_indices.begin(); it !=  clustered_indices.end(); it++,j++) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_pts(new pcl::PointCloud<pcl::PointXYZ>);
            for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
            {
                cluster_pts->points.push_back(input_cloud->points[*pit]);
            }
            cluster_pts->header = input_cloud->header;

            feature_extractor.setInputCloud(cluster_pts);
            feature_extractor.compute();
            pcl::PointXYZ min_point_AABB;
            pcl::PointXYZ max_point_AABB;
            feature_extractor.getAABB(min_point_AABB, max_point_AABB);


            float min_z = param.is_ext_obsv_3D ? min_point_AABB.z : 0.2;
            float max_z = param.is_ext_obsv_3D ? max_point_AABB.z : 2.6;
            float3 ll_coord{min_point_AABB.x, min_point_AABB.y, min_z};
            float3 ur_coord{max_point_AABB.x, max_point_AABB.y, max_z};
            ext_obs->append_new_elem(ll_coord, ur_coord);
        }
    }
}

void VOLMAPNODE::CB_ext_cld(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    clustring(msg);
}

void VOLMAPNODE::update_ext_map()
{
    float3 loc_map_ll = _loc_map->_msg_origin;
    float3 loc_map_ur  = make_float3(loc_map_ll.x + param.local_size_x,
                                     loc_map_ll.y + param.local_size_y,
                                     loc_map_ll.z + param.local_size_z);

    // construct ext obs map
    ext_obs->activate_AABB(loc_map_ll, loc_map_ur);

//    ext_obs->constructPreOGM(_og_map, _edt_map->_pvt, _time);


}
