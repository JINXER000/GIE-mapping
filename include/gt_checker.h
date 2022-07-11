

#ifndef SRC_GT_CHECKER_H
#define SRC_GT_CHECKER_H

#include <iostream>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>


class Gnd_truth_checker{
public:
    typedef pcl::PointCloud<pcl::PointXYZ> PntCld;
    typedef pcl::PointCloud<pcl::PointXYZI> PntCldI;
  Gnd_truth_checker()
  {
    occu_cld=PntCld::Ptr(new PntCld);
    edt_cld=PntCldI::Ptr(new PntCldI);
  }
  void cpy_occu_cld(PntCld::Ptr &cloud_in)
  {
    pcl::copyPointCloud(*cloud_in,*occu_cld);
  }
  void cpy_edt_cld(PntCldI::Ptr &cloud_in)
  {
    pcl::copyPointCloud(*cloud_in,*edt_cld);
  }
  float cmp_dist()
  {
    double ems1=0,ems2=0,max_err=0;
    int l_cnt=0,m_cnt=0;
    if(occu_cld->size()==0||edt_cld->size()==0)
    {
      printf("no checking due to empty cloud\n");
      occu_cld->clear();
      edt_cld->clear();
      return -1;
    }
    //    occu_cld->resize(occu_cld->height*occu_cld->width);
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(occu_cld);
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);

    for(int i=0;i<edt_cld->size();i++)
    {
      pcl::PointXYZI edt_pt=edt_cld->points[i];
      if(kdtree.nearestKSearch(pcl::PointXYZ(edt_pt.x, edt_pt.y, edt_pt.z), 1,
                               pointIdxNKNSearch, pointNKNSquaredDistance))
      {
        auto closest_obs=(*occu_cld)[pointIdxNKNSearch[0]];
        double knn_dist=sqrt(pointNKNSquaredDistance[0]);
        double edt_dist=edt_pt.intensity;
        double error=knn_dist-edt_dist;
        if(error>0.001) // EDT is less!
          l_cnt++;
        else if(error<-0.001)
          m_cnt++;
        ems1 += abs(error);
        ems2 += error * error;
        max_err = std::max(fabs(error), max_err);
      }

    }
    float rms_err=sqrt(ems2/edt_cld->size());
    rms_sum+=rms_err;
    rms_cnt++;
    if(rms_cnt>=10)
    {
      float rms_average=rms_sum/rms_cnt;
      printf("max_error is %f,  rms_err is %f\n",max_err,rms_average);
      rms_sum=0;
      rms_cnt=0;
    }
    occu_cld->clear();
    edt_cld->clear();
    return rms_err;
  }

public:
  PntCld::Ptr occu_cld;
  PntCldI::Ptr edt_cld;
  float rms_sum;
  int rms_cnt=0;
};
#endif //SRC_GT_CHECKER_H
