

# GIE-mapping

Source code for the paper: **GPU-accelerated Incremental Euclidean Distance Transform for Online Motion Planning of Mobile Robots**

This work has been accepted by IEEE Robotics and Automation Letters 2022 and was presented in IROS 2022, Kyoto.

This software is a volumetric mapping
system that effectively calculates Occupancy Grid Maps (OGMs)
and Euclidean Distance Transforms (EDTs) with GPU. 
Extensive experiments have been carried out with different robots and range sensors. The proposed system achieves real-time
performance with limited onboard computational resources.

<p align="center">
  <img src="figure/uav-2dlidar.gif" width = "400" height = "225"/>
  <img src="figure/uav-depth.gif" width = "400" height = "225"/>
  <img src="figure/uav-laser3d.gif" width = "400" height = "225"/>
  <img src="figure/ugv_laser3d.gif" width = "400" height = "225"/>
</p>



The supplementary video can be viewed here:

<p align="center">
<a href="https://youtu.be/1g4AnkHAiZ8
" target="_blank"><img src="figure/coverpage.png"
alt="GIE-mapping  introduction video" width="480" height="270" /></a>
</p>

## Supported data input:

- Any sensor outputs pointcloud (e.g., OS32 LiDAR, Mid360)
- Depth camera
- 2D LiDAR
- 3D LiDAR
- Priori knowledge 

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
1. This project runs CUDA and requires a computer with **Nvidia GPU**. We have successfully tested this project on CUDA 9.0, 10.2, 11.3 and 11.4.
2. Install Ubuntu with ROS. This project has been tested on Ubuntu 16.04 (ROS Kinetic), 18.04 (ROS Melodic) and 20.04 (ROS Noetic). 
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
sudo chmod 777 ./CMakeLists.txt
catkin_make
source devel/setup.bash
```


## Launch the mapper
### Caution
- If the local size is too large, an "invalid argument" error will be thrown due to CUDA does not support such a large thread-block.
- The parameters *bucket_max* and *block_max* has to be increased if you are doing large-scale and fine-resolution mapping. The initialization time may be longer. 
- The software is being actively updated, and there may be inconsisitency between our paper and the actual implementation. The most updated profiling data can be viewed in our [supplementary video](https://youtu.be/1g4AnkHAiZ8).


Please kindly leave a **star** if this software is helpful to your projects :3
## Try with datasets
### UGV-corridor
Please download the dataset [here](https://drive.google.com/file/d/1COHl_jEaWHl09kPolfXgYs66_YTrb3uH/view?usp=sharing).

```bash
roslaunch GIE ugv_dataset.launch
rosbag play ugv-cut-filter.bag --clock __name:=profile_bag
```

### Cow-lady
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
- Decrease the parameter *cutoff_dist* to a small number (e.g., 2m).
- Turn on *fast_mode* parameter. It will disable wavefront A and wavefront B (please see details in our [paper](https://ieeexplore.ieee.org/abstract/document/9782137)). If working in confined space, (e.g., Cow-Lady dataset), the accuracy is nearly the same as the original mode.

### Integrate with motion planners
Please set *for_motion_planner* parameter as true. It makes the current robot position valid and observed.

Our system publishes the EDT surround by the robot as CostMap.msg in the topic "cost_map". Each voxel contains visibility information and the distance value. If your motion planning package are not implemented together with GIE, then you can only access the local EDT information by subscribing to the topic "cost_map".

To access the global EDT directly, you are recommended to implement a  GPU-based motion planner together with GIE-mapping. 
Each voxel  can be retrieved by using device function  *get_VB_key()* and *get_voxID_in_VB()*. In this way, the *display_glb_ogm* parameter can be *false*, saveing you tons of time.

If you are using a  CPU-based planner, you can retrieve the voxel block ID like this:
```cpp
 int VB_idx =_hash_map->hash_table_H_std.find(blk_key)->second;
```
And voxels inside the block can be visited with *get_voxID_in_VB()*.

# Additional features
### Frontiers for exploration
The system extracts low-level frontiers for exploration. The data type of **VOXTYPE_FNT** denotes the voxel that belongs to the low-level frontier. You may need to do some post-process to filter out the noise. 

### Signed distance 
Developing

### Virtual fence 
Revise the prebuilt map in parameters.h. Note that obsbbx_ll and obsbbx_ur are the lower-left corner and upper-right corner of the flyable region. 

### External observer:
Publish a pointcloud in topic "forbid_reg_cloud".
If the pointcloud is 3D, please set **is_ext_obsv_3D** as true. Otherwise, the height of external observed obstacle is 0.2m~2.6m.

# Docker Support

To make the installation and development of GIE-mapping easier, the docker-based installation is introduced for Nvidia Xavier NX. Users only need to pull the image from [docker hub](hub.docker.com), and use the provided scripts to finish the installation easily.

## Docker installation
Users can follow the simplified instructions below or follow the [Official instruction](https://docs.docker.com/engine/install/ubuntu/) and [Post-installation steps for Linux](https://docs.docker.com/engine/install/linux-postinstall/).
1. Downloads the script from get.docker.com and runs it to install the latest stable release of Docker on Linux
    ```bash
    curl -fsSL get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    ```
2. Create the docker group
    ```bash
    sudo groupadd docker
    ```
3. Add your user to the docker group
    ```bash
    sudo usermod -aG docker $USER
    ```
4. Log out and log back in so that your group membership is re-evaluated.

    On a desktop Linux environment such as X Windows, log out of your session completely and then log back in.

    On Linux, you can also run the following command to activate the changes to groups:
    ```bash
    newgrp docker 
    ```

5. Verify that you can run docker commands without sudo
    ```bash
    docker run hello-world
    ```
## GIE-mapping installation
Thanks to the properties of Docker, the installation is very straightforward. Users only need to clone the installation script from the repo and run it in Host BASH and Docker BASH respectively. Then, all the installation steps will be finished.

```bash
git clone https://github.com/ryrobotics/USR_Docker.git
mkdir -p ~/GIE_src
cp ./USR_Docker/ROS-melodic-Xavier/GIE_Dokcer_Install.sh ~/GIE_src/
cd ~/GIE_src
sh GIE_Dokcer_Install.sh host
# Run in the bash of Docker
sh /src/GIE_Dokcer_Install.sh docker
```

# Trouble shooting
As reported in  [issue 1](https://github.com/JINXER000/GIE-mapping/issues/1), there might be some problems in launching the mapper with Ubuntu 20.04. Please ensure that the GPU model, GPU driver version, and CUDA version match with each other. For more details, you can refer to [this website](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/). And it is recommend to activate C++ 14 in CMake when using 20.04. 


# Change log
### Feb 20, 2023
- Flight success with Mid360
- Prior knowledge is partially supported
### Oct 21, 2022
- Add Docker support
- Remove hard-code logging directory. 
### June 24, 2022
- Add support for 16-line 3D LiDAR.
- Fix some bugs.
### July 11, 2022
- Batch EDT is largely accelerated.
- Fix some bugs in corner cases.
- Add Fast mode.