

# GIE-mapping

Source code for the paper: **GPU-accelerated Incremental Euclidean Distance Transform for Online Motion Planning of Mobile Robots**

This work has been accepted by IEEE Robotics and Automation Letters 2022. 

This software is a volumetric mapping
system that effectively calculates Occupancy Grid Maps (OGMs)
and Euclidean Distance Transforms (EDTs) with GPU. 
Extensive experiments have been carried out with different robots and range sensors. The proposed system achieves real-time
performance with limited onboard computational resources.

<p align="center">
  <img src="figure/uav-2dlidar.gif" width = "480" height = "270"/>
  <img src="figure/uav-depth.gif" width = "480" height = "270"/>
  <img src="figure/uav-laser3d.gif" width = "480" height = "270"/>
  <img src="figure/ugv_laser3d.gif" width = "480" height = "270"/>
</p>



The supplementary video can be viewed here:

<p align="center">
<a href="https://youtu.be/ne9PZtcLoGc
" target="_blank"><img src="figure/coverpage.png"
alt="GIE-mapping  introduction video" width="480" height="270" /></a>
</p>

## Supported sensors:

- Any sensor outputs pointcloud
- Depth camera
- 2D LiDAR
- 3D LiDAR


Please cite our paper if you use this project in your research:

```
@ARTICLE{9782137,
  author={Chen, Yizhou and Lai, Shupeng and Cui, Jinqiang and Wang, Biao and Chen, Ben M.},
  journal={IEEE Robotics and Automation Letters}, 
  title={GPU-Accelerated Incremental Euclidean Distance Transform for Online Motion Planning of Mobile Robots}, 
  year={2022},
  volume={7},
  number={3},
  pages={6894-6901},
  doi={10.1109/LRA.2022.3177852}}
 ````

 The current implementation of voxel hashing references [this repo](https://github.com/xkjyeah/vhashing).
 We plan to improve the efficiency of this part by adopting more efficient data structures of GPU hash table.
# Installation
## Prerequisite
1. This project runs CUDA and requires a computer with **Nvidia GPU**. We have successfully tested this project on CUDA 9.0, 10.2, and 11.1.
2. Install Ubuntu with ROS. This project has been tested on Ubuntu 16.04(ROS Kinetic) and 18.04(ROS Melodic). 
## Recompile cuTT 
cuTT is a library used for faster batch EDT. 
````bash
git clone https://gitee.com/jinxer000/cutt_lts.git
cd cutt_lts
rm ./build/*
make 
````
It will create  the library itself:
- include/cutt.h
- lib/libcutt.a
 
Copy the lib file into $(GIE_folder)/lib.

If it fails to compile, please modify the Makefile according to [this website](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).

## Seperate compilation lib
Search your computer and find the *libcudadevrt.a* (e.g., /usr/local/cuda-10.2/targets/x86_64-linux/lib/libcudadevrt.a).

Copy the lib file into $(GIE_folder)/lib.

## Compile 
Download source code from github:
```bash
mkdir GIE_ws & cd GIE_ws
mkdir src & cd src
git clone https://github.com/JINXER000/GIE-mapping.git
cd ..
catkin_make
source devel/setup.bash
```
Before launching any examples, please revise the *log_dir* parameter in the corresponding .yaml file
to your own directory, otherwise an error will be thrown. 

## Launch the mapper
## Try with datasets
### UGV-corridor
Please download the dataset [here](https://drive.google.com/file/d/1COHl_jEaWHl09kPolfXgYs66_YTrb3uH/view?usp=sharing).

```bash
roslaunch GIE ugv_dataset.launch
rosbag play ugv-cut-filter.bag --clock __name:=profile_bag
```

### cow-lady
Please download the dataset [here](http://robotics.ethz.ch/~asl-datasets/iros_2017_voxblox/data.bag).
```bash
roslaunch GIE cow_dataset.launch
rosbag play data.bag --clock __name:=profile_bag
```

### UAV-2DLiDAR
Please download the dataset [here](https://drive.google.com/file/d/1ns8DkFRKN-9JuG-FQcYzLzghA18B7m4c/view?usp=sharing).

```bash
roslaunch GIE scan2D.launch
rosbag play uav-2dlidar-cut-filter.bag --clock 
```

### UAV-depthcam
Please download the dataset [here](https://drive.google.com/file/d/16M_smQXJOCzElDDeMadCClYup6jEZ9ec/view?usp=sharing).
```bash
roslaunch GIE depthcam_iris.launch
rosbag play uav-depth.bag --clock 
```
### UAV-3DLiDAR
Please download the dataset [here](https://www.dropbox.com/s/08vjsaqw6c1ppmv/hector_straight.bag?dl=0).

Use volumetric projection:
```bash
roslaunch GIE uav_laser3d.launch
rosbag play hector_straight.bag --clock 
```

Use parallel ray casting:
```bash
roslaunch GIE uav_raycast.launch
rosbag play hector_straight.bag --clock 
```

## Try on your own robot
Simply remap the input data in volumetric_mapper.cpp to your own sensor topics!

Remember to set *use_sim_time* parameter in each launch file as **false** in the real world.

### Speed up tricks
- Turn off Rviz during the run since it will occupy a large amount of GPU resources.
- Disable both *display_glb_edt* and *display_glb_ogm* parameter. Hence the GPU hash table won't be streamed to CPU at every iteration.
- Decrease the parameter *cutoff_dist* to a small number (e.g., 2).

### Integrate with motion planners
Our system publishes the EDT surround as CostMap.msg in the topic "cost_map". Each voxel contains visibility information and the distance value. If your motion planning package are not implemented together with GIE, then you can only access the local EDT information by subscribing to the topic "cost_map".

To access the global EDT directly, you are recommended to implement a  GPU-based motion planner together with GIE-mapping. 
Each voxel  can be retrieved by using device function  *get_VB_key()* and *get_voxID_in_VB()*. 

If you are using a  CPU-based planner, the voxel block ID can be retrieved like this:
```cpp
 int VB_idx =_hash_map->hash_table_H_std.find(blk_key)->second;
```
And voxels inside the block can be visited.

### Caution
- If the local size is too large, an "invalid argument" error will be thrown due to CUDA does not support such a large thread-block.

- The parameters *bucket_max* and *block_max* has to be increased if you are doing large-scale and fine-resolution mapping. The initialization time may be longer. 
# Additional features
### Frontiers for exploration
The system extracts low-level frontiers for exploration. The data type of **VOXTYPE_FNT** denotes the voxel that belongs to the low-level frontier. You may need to do some post-process to filter out the noise. 

### Signed distance 
Developing


# Change log
### June 24, 2022
- Add support for 16-line 3D LiDAR.
- Fixed some bugs.