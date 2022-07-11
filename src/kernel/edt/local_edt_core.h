

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
            loc_map.g(c.x,c.y,c.z,0) = 0;
            int3 coc_xyz = int3{c.x,c.y,c.z};
            loc_map.coc_idx(c.x,c.y,c.z,0) = loc_map.loc_coc2idx(coc_xyz);
        }
        else
        {
            loc_map.g(c.x,c.y,c.z,0) = loc_map._max_width;
            int3 coc_xyz = loc_map.INVALID_LOC_COC;
            loc_map.coc_idx(c.x,c.y,c.z,0) = loc_map.loc_coc2idx(coc_xyz);
        }

        for (c.y=1; c.y<loc_map._local_size.y; c.y++)
        {
            vox_type = loc_map.get_vox_glb_type(c);
            if (vox_type == VOXTYPE_OCCUPIED)
            {
                loc_map.g(c.x,c.y,c.z,0) = 0;
                int3 coc_xyz = int3{c.x,c.y,c.z};
                loc_map.coc_idx(c.x, c.y,c.z,0) = loc_map.loc_coc2idx(coc_xyz);
            }
            else
            {
                int3 coc_xymz = loc_map.get_coc_viaID(c.x, c.y-1, c.z, 0, false);
                int3 coc_xyz = loc_map.get_coc_viaID(c.x, c.y, c.z, 0, false);
                if(coc_xymz.y< loc_map._max_width)
                {
                    loc_map.g(c.x,c.y,c.z,0) = 1 + loc_map.g(c.x,c.y-1,c.z,0);
                    coc_xyz.y =  coc_xymz.y;
                    loc_map.coc_idx(c.x, c.y, c.z, 0) = loc_map.loc_coc2idx(coc_xyz);
                }else
                {
                    loc_map.g(c.x,c.y,c.z,0)=loc_map._max_width;
                    int3 coc_xyz = loc_map.INVALID_LOC_COC;
                    loc_map.coc_idx(c.x, c.y,c.z,0) = loc_map.loc_coc2idx(coc_xyz);
                }
            }
        }
        // Second pass
        for (c.y=loc_map._local_size.y-2;c.y>=0;c.y--)
        {
            int3 coc_xypz = loc_map.get_coc_viaID(c.x, c.y+1, c.z, 0, false);
            int3 coc_xyz = loc_map.get_coc_viaID(c.x, c.y, c.z, 0, false);
            if (loc_map.g(c.x,c.y+1,c.z,0) < loc_map.g(c.x,c.y,c.z,0))
            {
                if(coc_xypz.y < loc_map._max_width)
                {
                    loc_map.g(c.x,c.y,c.z,0) = 1+loc_map.g(c.x,c.y+1,c.z,0);
                    coc_xyz.y =  coc_xypz.y;
                    loc_map.coc_idx(c.x, c.y, c.z, 0) = loc_map.loc_coc2idx(coc_xyz);
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
        for (int u=1;u<loc_map._local_size.x;u++)
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
                if (w < loc_map._local_size.x)
                {
                    q++;
                    loc_map.s(x,q,z,1)=u;
                    loc_map.t(x,q,z,1)=w;
                }
            }
        }
        for (int u=loc_map._local_size.x-1;u>=0;u--) // coc_aux <- coc
        {
            loc_map.g(x,u,z,1)=loc_map.f(u,loc_map.s(x,q,z,1),x,z);

            int tmpcocx=loc_map.s(x,q,z,1);
            int3 coc_xuz = loc_map.get_coc_viaID(x, u, z, 1, false);
            coc_xuz.x = tmpcocx;

            int3 coc_xtxz_aux = loc_map.get_coc_viaID(x, tmpcocx, z, 1, true);
            if(coc_xtxz_aux.y< loc_map._max_width)
            {
                coc_xuz.y = coc_xtxz_aux.y;
            }
            loc_map.coc_idx(x, u, z, 1) = loc_map.loc_coc2idx(coc_xuz);

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
        for (int u=1; u<loc_map._local_size.z; u++)
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
                if (w<loc_map._local_size.z)
                {
                    q++;
                    loc_map.s(x,q,z,2) = u;
                    loc_map.t(x,q,z,2) = w;
                }
            }
        }
        for (int u=loc_map._local_size.z-1;u>=0;u--)
        {
            loc_map.g(x,u,z,2) = loc_map.f_z(u,loc_map.s(x,q,z,2),z,x);

            int tmpcocz=loc_map.s(x,q,z,2);
            int3 coc_xuz = loc_map.get_coc_viaID(x, u, z, 2, false);
            coc_xuz.z = tmpcocz;

            int3 coc_xtzz_aux = loc_map.get_coc_viaID(x, tmpcocz, z, 2, true);
            if(coc_xtzz_aux.x< loc_map._max_width)
            {
                coc_xuz.x = coc_xtzz_aux.x;
            }

            if(coc_xtzz_aux.y< loc_map._max_width)
            {
                coc_xuz.y = coc_xtzz_aux.y;
            }
            loc_map.coc_idx(x, u, z, 2) = loc_map.loc_coc2idx(coc_xuz);

            if (u==loc_map.t(x,q,z,2))
                q--;

        }
    }

}
#endif //SRC_LOCAL_EDT_CORE_H
