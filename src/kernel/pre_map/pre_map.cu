
#include "map_structure/pre_map.h"


Ext_Obs_Wrapper::Ext_Obs_Wrapper(int obs_num, int free_num)
{
    change_obs_num(obs_num);
    change_free_num(free_num);
}

void Ext_Obs_Wrapper::assign_obs_premap(std::vector<float3>& preobs_ll, std::vector<float3>& preobs_ur)
{
    rt_obsbbx_ll.assign(preobs_ll.begin(), preobs_ll.end());
    rt_obsbbx_ur.assign(preobs_ur.begin(), preobs_ur.end());
}

void Ext_Obs_Wrapper::assign_free_premap(std::vector<float3>& preFree_ll, std::vector<float3>& preFree_ur)
{

    // assign
    rt_freeBBX_ll.assign(preFree_ll.begin(), preFree_ll.end());
    rt_freeBBX_ur.assign(preFree_ur.begin(), preFree_ur.end());
}


float euclidean_dist(float3& obs1, float3& obs2)
{
    float3 diff = make_float3(obs1.x-obs2.x, obs1.y-obs2.y, obs1.z-obs2.z);
    return length(diff);
}

void Ext_Obs_Wrapper::append_new_elem(float3& ll_coord, float3& ur_coord)
{
//    float3 obs_center = make_float3((ll_coord.x+ ur_coord.x)*0.5,
//                                    (ll_coord.y+ ur_coord.y)*0.5,
//                                    (ll_coord.z+ ur_coord.z)*0.5);
//
//    bool existed = false;
//    for (int i=0; i< ext_obs_num; i++)
//    {
//        float3 existed_obs_center = make_float3((rt_obsbbx_ll[i].x+ rt_obsbbx_ur[i].x)*0.5,
//                                                (rt_obsbbx_ll[i].y+ rt_obsbbx_ur[i].y)*0.5,
//                                                (rt_obsbbx_ll[i].z+ rt_obsbbx_ur[i].z)*0.5);
//        float dist =euclidean_dist(obs_center, existed_obs_center);
//        if (dist < 0.3)
//        {
//            existed = true;
//            break;
//        }
//    }

//    if(!existed)
    {
        rt_obsbbx_ll.push_back(ll_coord);
        rt_obsbbx_ur.push_back(ur_coord);
    }

}

// return: if there is obs in pre-map.
bool Ext_Obs_Wrapper::bbx_H2D()
{
    if (rt_obsbbx_ll.size() == 0)
    {
        return false;
    }

    if(ext_obs_num != rt_obsbbx_ll.size())
    {
        change_obs_num(rt_obsbbx_ll.size());
    }

    thrust::copy(rt_obsbbx_ll.begin(), rt_obsbbx_ll.end(), obsbbx_ll_D.begin());
    thrust::copy(rt_obsbbx_ur.begin(), rt_obsbbx_ur.end(), obsbbx_ur_D.begin());

    if (rt_freeBBX_ll.size() == 0)
    {
        return true;
    }
    if(ext_free_num != rt_freeBBX_ll.size())
    {
        change_free_num(rt_freeBBX_ll.size());
    }
    thrust::copy(rt_freeBBX_ll.begin(), rt_freeBBX_ll.end(), freeBBX_ll_D.begin());
    thrust::copy(rt_freeBBX_ur.begin(), rt_freeBBX_ur.end(), freeBBX_ur_D.begin());

    return true;
}

void Ext_Obs_Wrapper::change_obs_num(int obs_num)
{
    ext_obs_num = obs_num;
    // change device size
    obsbbx_ll_D.resize(ext_obs_num);
    obsbbx_ur_D.resize(ext_obs_num);
    obs_activated.resize(ext_obs_num);

}

void Ext_Obs_Wrapper::change_free_num(int free_num)
{
    // init
    ext_free_num = free_num;
    freeBBX_ll_D.resize(ext_free_num);
    freeBBX_ur_D.resize(ext_free_num);
}

bool Ext_Obs_Wrapper::CheckAABBIntersection(float3& A_ll, float3& A_ur, float3& B_ll, float3& B_ur)
{
    bool x_overlap = (A_ll.x <= B_ur.x) && (A_ur.x>=B_ll.x);
    bool y_overlap = (A_ll.y <= B_ur.y) && (A_ur.y>=B_ll.y);
    bool z_overlap = (A_ll.z <= B_ur.z) && (A_ur.z>=B_ll.z);
    return (x_overlap && y_overlap && z_overlap);
}



void Ext_Obs_Wrapper::activate_AABB(float3& loc_map_ll, float3& loc_map_ur)
{
    bool require_activate = bbx_H2D();

    if (!require_activate)
    {
        return;
    }
    // outmost bbx
    obs_activated[0] = false;

    // tell if one AABB is to be checked
    for (int i=1; i< ext_obs_num; i++)
    {
        bool is_intersect = CheckAABBIntersection(loc_map_ll, loc_map_ur,
                                                  rt_obsbbx_ll[i], rt_obsbbx_ur[i]);
        if (is_intersect)
        {
            obs_activated[i] = true;
        }else
        {
            obs_activated[i] = false;
        }

    }

}