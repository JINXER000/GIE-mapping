#ifndef SRC_LOCAL_EDT_CORE_H
#define SRC_LOCAL_EDT_CORE_H

#include <cuda_toolkit/projection.h>
#include "map_structure/local_batch.h"
#include <cutt/cutt.h>
#include "par_wave/voxmap_utils.cuh"

namespace EDT_CORE
{

    __global__
    void EDTphase1(LocMap loc_map, int time)
    {
        int3 c;
        c.z = blockIdx.x;
        c.x = threadIdx.x;
        c.y = 0;


        // First pass
        int vox_type = loc_map.get_vox_glb_type(c);
        if (vox_type == VOXTYPE_OCCUPIED)
        {
            loc_map.g(c.x,0,c.z,0) = 0;
            loc_map.coc(c.x, c.y, c.z) = int3{c.x,c.y,c.z};
        }
        else
        {
            loc_map.g(c.x,0,c.z,0) = loc_map._max_width;
            loc_map.coc(c.x,c.y,c.z) = EMPTY_KEY;
        }
        loc_map.coc_aux(c.x, c.y, c.z) = EMPTY_KEY;

        for (c.y=1; c.y<loc_map._update_size.y; c.y++)
        {
            vox_type = loc_map.get_vox_glb_type(c);
            if (vox_type == VOXTYPE_OCCUPIED)
            {
                loc_map.g(c.x,c.y,c.z,0) = 0;
                loc_map.coc(c.x, c.y, c.z) = int3{c.x,c.y,c.z};
            }
            else
            {

                if (loc_map.coc(c.x, c.y-1, c.z).y < loc_map._max_width)
                {
                    loc_map.g(c.x,c.y,c.z,0) = 1 + loc_map.g(c.x,c.y-1,c.z,0);
                    loc_map.coc(c.x,c.y,c.z).y=loc_map.coc(c.x,c.y-1,c.z).y;
                }else
                {
                    loc_map.g(c.x,c.y,c.z,0)=loc_map._max_width;
                    loc_map.coc(c.x,c.y,c.z) = EMPTY_KEY;
                }

            }
            loc_map.coc_aux(c.x, c.y, c.z) = EMPTY_KEY;
        }
        // Second pass
        for (c.y=loc_map._update_size.y-2;c.y>=0;c.y--)
        {
            if (loc_map.g(c.x,c.y+1,c.z,0) < loc_map.g(c.x,c.y,c.z,0))
            {
                if (loc_map.coc(c.x, c.y+1, c.z).y < loc_map._max_width)
                {
                    loc_map.g(c.x,c.y,c.z,0) = 1+loc_map.g(c.x,c.y+1,c.z,0);
                    loc_map.coc(c.x,c.y,c.z).y=loc_map.coc(c.x,c.y+1,c.z).y;
                }else
                {
                    loc_map.g(c.x,c.y,c.z,0) = loc_map._max_width;
                }
            }

        }

    }

    __global__
    void EDTphase2(LocMap loc_map)
    {
        int z = blockIdx.x;
        int x = threadIdx.x;

        int q=0;
        loc_map.s(x,0,z,1) = 0;
        loc_map.t(x,0,z,1) = 0;
        for (int u=1;u<loc_map._update_size.x;u++)
        {
            while (q>=0 && loc_map.f(loc_map.t(x,q,z,1),loc_map.s(x,q,z,1),x,z)
                           > loc_map.f(loc_map.t(x,q,z,1),u,x,z))
            {
                q--;
            }
            if (q<0)
            {
                q = 0;
                loc_map.s(x,0,z,1)=u;
            }
            else
            {
                int w = 1+loc_map.sep(loc_map.s(x,q,z,1),u,x,z);
                if (w < loc_map._update_size.x)
                {
                    q++;
                    loc_map.s(x,q,z,1)=u;
                    loc_map.t(x,q,z,1)=w;
                }
            }
        }
        for (int u=loc_map._update_size.x-1;u>=0;u--)
        {
            loc_map.g(x,u,z,1)=loc_map.f(u,loc_map.s(x,q,z,1),x,z);

            int tmpcocx=loc_map.s(x,q,z,1);
            loc_map.coc_aux(u,x,z).x=tmpcocx;

            int tmpcocy=loc_map.coc(tmpcocx,x,z).y;
            if(tmpcocy < loc_map._max_width)
                loc_map.coc_aux(u,x,z).y=tmpcocy;

            if (u == loc_map.t(x,q,z,1))
                q--;

        }
    }

    __global__
    void EDTphase3(LocMap loc_map)
    {
        int z = blockIdx.x;
        int x = threadIdx.x;

        int q = 0;
        loc_map.s(x,0,z,2) = 0;
        loc_map.t(x,0,z,2) = 0;
        for (int u=1; u<loc_map._update_size.z; u++)
        {
            while (q>=0 && loc_map.f_z(loc_map.t(x,q,z,2),loc_map.s(x,q,z,2),z,x)
                           > loc_map.f_z(loc_map.t(x,q,z,2),u,z,x))
            {
                q--;
            }
            if (q<0)
            {
                q = 0;
                loc_map.s(x,0,z,2) = u;
            }
            else
            {
                int w = 1+loc_map.sep_z(loc_map.s(x,q,z,2),u,z,x);
                if (w<loc_map._update_size.z)
                {
                    q++;
                    loc_map.s(x,q,z,2) = u;
                    loc_map.t(x,q,z,2) = w;
                }
            }
        }
        for (int u=loc_map._update_size.z-1;u>=0;u--)
        {
            loc_map.g(x,u,z,2) = loc_map.f_z(u,loc_map.s(x,q,z,2),z,x);

            int tmpcocz=loc_map.s(x,q,z,2);
            loc_map.coc(z,x,u).z=tmpcocz;

            int tmpcocx=loc_map.coc_aux(z,x,tmpcocz).x;
            if(tmpcocx < loc_map._max_width)
                loc_map.coc(z,x,u).x=tmpcocx;
            int tmpcocy=loc_map.coc_aux(z,x,tmpcocz).y;
            if(tmpcocy< loc_map._max_width)
                loc_map.coc(z,x,u).y=tmpcocy;

            if (u==loc_map.t(x,q,z,2))
                q--;

        }
    }

}
#endif //SRC_LOCAL_EDT_CORE_H
