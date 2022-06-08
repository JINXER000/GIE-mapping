#include "volumetric_mapper.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vol_node");
    VOLMAPNODE vol_mapper;
    ros::spin();

    return 0;
}
