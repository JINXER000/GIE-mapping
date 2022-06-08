#ifndef PROJECTION_H
#define PROJECTION_H
#include <cuda_toolkit/cuda_macro.h>
#include <nav_msgs/Odometry.h>
#include <tf/tf.h>
struct Projection
{
    cudaMat::SE3<float> L2G;
    cudaMat::SE3<float> G2L;
    float3 origin;
};


__forceinline__
Projection trans2proj(const tf::Transform &trans)
{

    tf::Quaternion Rotq=trans.getRotation();
    tf::Vector3 transVec=trans.getOrigin();

    Projection proj;
    proj.L2G = cudaMat::SE3<float>(static_cast<float>(Rotq.w()),static_cast<float>(Rotq.x()),
                                   static_cast<float>(Rotq.y()),static_cast<float>(Rotq.z()),
                                   static_cast<float>(transVec.m_floats[0]),
                                   static_cast<float>(transVec.m_floats[1]),
                                   static_cast<float>(transVec.m_floats[2]));
    proj.G2L = proj.L2G.inv();
    proj.origin.x = static_cast<float>(trans.getOrigin().x());
    proj.origin.y = static_cast<float>(trans.getOrigin().y());
    proj.origin.z = static_cast<float>(trans.getOrigin().z());

    return proj;
}
#endif // PROJECTION_H
