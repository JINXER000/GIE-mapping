cmake_minimum_required(VERSION 2.8.3)
project(GIE)

# Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    nav_msgs
    pcl_conversions
    pcl_ros
    tf_conversions
    )

FIND_PACKAGE(CUDA REQUIRED)

set(CUDA_SEPARABLE_COMPILATION TRUE)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
            "MinSizeRel" "RelWithDebInfo")
endif()

if( ${CMAKE_BUILD_TYPE} STREQUAL Release)
    SET(CUDA_NVCC_FLAGS "-arch=sm_61;-O3;-use_fast_math;-ftz=true;-prec-div=false;-prec-sqrt=false;-rdc=true" CACHE STRING "nvcc flags" FORCE)
    message("build CUDA with -O3")
else()
    message("build CUDA with Debug")
    SET(CUDA_NVCC_FLAGS "-g;-G;-arch=sm_61;-ftz=true;-rdc=true" CACHE STRING "nvcc flags" FORCE)
    SET(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
endif()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)



add_message_files(
    FILES
    CostMap.msg
    )

generate_messages(
    DEPENDENCIES
    std_msgs
    )

catkin_package(
    INCLUDE_DIRS include ${CUDA_INCLUDE_DIRS}
#    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
    #  DEPENDS system_lib
    )

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    include/par_wave
    ${catkin_INCLUDE_DIRS}
    )

LINK_DIRECTORIES(
    lib
    )


cuda_add_executable(
    ${PROJECT_NAME}_node
    src/main.cpp
    src/kernel/hokuyo/hokuyo_fast.cu
    src/kernel/realsense/realsense_fast.cu
    src/kernel/point_cloud/pntcld_raycast.cu
    src/hokuyo_map_maker.cpp
    src/realsense_map_maker.cpp
    src/pntcld_map_maker.cpp
    src/kernel/par_wave/glb_hash_map.cu
    src/volumetric_mapper.cpp
    src/kernel/edt/local_edt.cu src/vlp16_map_maker.cpp
    src/kernel/vlp16/vlp16_fast.cu
)

add_dependencies(
    ${PROJECT_NAME}_node
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
    )

## Specify libraries to link a library or executable target against
target_link_libraries(
        ${PROJECT_NAME}_node
        ${catkin_LIBRARIES}
        ${CUDA_LIBRARIES}
)
target_link_libraries(${PROJECT_NAME}_node cutt cudadevrt)

