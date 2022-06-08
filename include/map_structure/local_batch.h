#ifndef SRC_LOCAL_BATCH_H
#define SRC_LOCAL_BATCH_H

#include <cuda_toolkit/cuda_macro.h>
#define VOXTYPE_OCCUPIED 2
#define VOXTYPE_FREE 1
#define VOXTYPE_UNKNOWN 0
#define VOXTYPE_FNT 3


struct SeenDist
{
    float d; // distance
    bool s;  // seen
    bool o; // used for thining algorithm
};


class LocMap
{
public:
    LocMap(const float voxel_size, const int3 update_size,const unsigned char occupancy_threshold,
           const float ogm_min_h, const float ogm_max_h, const int cutoff_grids_sq):
            _voxel_width(voxel_size),
            _update_size(update_size),
            _occu_thresh(occupancy_threshold),
            _update_max_h(ogm_max_h),
            _update_min_h(ogm_min_h),
            _cutoff_grids_sq(cutoff_grids_sq)
    {
        _map_volume = update_size.x * update_size.y * update_size.z;
        _max_width = update_size.x + update_size.y + update_size.z;
        _max_loc_dist_sq = update_size.x*update_size.x + update_size.y*update_size.y + update_size.z*update_size.z;
        _bdr_num =  2*(update_size.x * update_size.y + update_size.y* update_size.z + update_size.x*update_size.z);
        _half_shift = make_int3(_update_size.x/2, _update_size.y/2, _update_size.z/2);
        seendist_size = _map_volume*sizeof(SeenDist);
    }

    void create_gpu_map()
    {
        GPU_MALLOC(&_ray_count, _map_volume*sizeof(int));
        GPU_MEMSET(_ray_count, 0, _map_volume*sizeof(int));
        GPU_MALLOC(&_inst_type, _map_volume*sizeof(char));
        GPU_MEMSET(_inst_type, 0, _map_volume*sizeof(char));
        GPU_MALLOC(&_glb_type, _map_volume*sizeof(char));
        GPU_MEMSET(_glb_type, 0, _map_volume*sizeof(char));
        GPU_MALLOC(&_edt_D, _map_volume*sizeof(float));
        GPU_MEMSET(_edt_D, 0, _map_volume*sizeof(float));


        GPU_MALLOC(&_aux, _map_volume*sizeof(int));
        GPU_MALLOC(&_g, _map_volume*sizeof(int));
        GPU_MALLOC(&_s, _map_volume*sizeof(int));
        GPU_MALLOC(&_t, _map_volume*sizeof(int));
        GPU_MALLOC(&_coc, sizeof(int3)*_map_volume);
        GPU_MALLOC(&_aux_coc, sizeof(int3)*_map_volume);

        glb_type_H = new char[_map_volume];
        edt_H = new float[_map_volume];
        seendist_out = new SeenDist[_map_volume];
    }

    void delete_gpu_map()
    {
        GPU_FREE(_ray_count);
        GPU_FREE(_inst_type);
        GPU_FREE(_glb_type);
        GPU_FREE(_edt_D);
//        GPU_FREE(_reged);
        GPU_FREE(_g);
        GPU_FREE(_s);
        GPU_FREE(_t);
        GPU_FREE(_aux);
        GPU_FREE(_coc);
        GPU_FREE(_aux_coc);

        delete [] glb_type_H;
        delete [] edt_H;
        delete [] seendist_out;
    }


    __device__
    bool is_inside_update_volume(const int3 & crd) const
    {
        if (crd.x<0 || crd.x>=_update_size.x ||
            crd.y<0 || crd.y>=_update_size.y ||
            crd.z<0 || crd.z>=_update_size.z)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    __host__
    int3 calculate_pivot_origin(float3 map_center)
    {
        int3 pivot = pos2coord(map_center);
        pivot.x -= _update_size.x/2;
        pivot.y -= _update_size.y/2;
        pivot.z -= _update_size.z/2;

        float3 origin = coord2pos(pivot);

        // set pvt
        _pvt = pivot;
        _msg_origin = origin;
        return pivot;
    }

    __device__ __host__
    int coord2idx_local(const int3 &c) const
    {
        return c.x + c.y*_update_size.x + c.z*_update_size.x*_update_size.y;
    }

    __host__ __device__
    int3 pos2coord(const float3 & p) const
    {
        int3 output;

        output.x = floorf( p.x/ _voxel_width + 0.5f);
        output.y = floorf( p.y/ _voxel_width + 0.5f);
        output.z = floorf( p.z/ _voxel_width + 0.5f);
        return output;
    }
    __host__ __device__
    float3 coord2pos(const int3 & c) const
    {
        float3 output;
        output.x = c.x * _voxel_width;
        output.y = c.y * _voxel_width;
        output.z = c.z * _voxel_width;
        return output;
    }
    __host__ __device__
    int3 glb2loc(const int3 & c) const
    {
        return c-_pvt;
    }

    __host__ __device__
    int3 loc2glb(const int3 & c) const
    {
        return c+_pvt;
    }
    __device__
    int atom_add_type_count(const int3 &crd, int val)
    {
        if (!is_inside_update_volume(crd))
            return -1;

#ifdef __CUDA_ARCH__
        return atomicAdd(&(_ray_count[coord2idx_local(crd)]), val);
#endif
    }
    __device__
    int get_vox_count(const int3 &crd)
    {
        if (!is_inside_update_volume(crd))
            return 0;

        int idx = coord2idx_local(crd);
        return _ray_count[idx];
    }

    __device__
    void set_vox_count(const int3 &crd, int val)
    {
        if (!is_inside_update_volume(crd))
            return;

        int idx = coord2idx_local(crd);
        _ray_count[idx]= val;
    }
    __device__
    char get_vox_type(const int3 &crd)
    {
        if  (!is_inside_update_volume(crd))
            return VOXTYPE_UNKNOWN;

        int idx = coord2idx_local(crd);
        return _inst_type[idx];
    }

    __device__
    void set_vox_type(const int3 &crd, const char type)
    {
        if  (!is_inside_update_volume(crd))
            return;

        int idx = coord2idx_local(crd);
        _inst_type[idx] = type;
    }
    __device__
    char get_vox_glb_type(const int3 &crd)
    {
        if  (!is_inside_update_volume(crd))
            return VOXTYPE_UNKNOWN;

        int idx = coord2idx_local(crd);
        return _glb_type[idx];
    }

    __device__
    void set_vox_glb_type(const int3 &crd, const char type)
    {
        if  (!is_inside_update_volume(crd))
            return;

        int idx = coord2idx_local(crd);
        _glb_type[idx] = type;
    }

    void copy_ogm_2_host()
    {
        GPU_MEMCPY_D2H(glb_type_H,_glb_type,sizeof(char)*_map_volume);
    }

    void copy_edt_2_host()
    {
        GPU_MEMCPY_D2H(edt_H,_edt_D,sizeof(float)*_map_volume);
    }

    __device__
    int get_bdr_idx(const int& face,const int3& cur_buf_crd)
    {

        switch (face) {
            case 0:
                return cur_buf_crd.z*_update_size.y+cur_buf_crd.y;
            case 1:
                return cur_buf_crd.z*_update_size.y+cur_buf_crd.y+_update_size.y*_update_size.z;
            case 2:
                return cur_buf_crd.z*_update_size.x+cur_buf_crd.x+2*_update_size.y*_update_size.z;
            case 3:
                return cur_buf_crd.z*_update_size.x+cur_buf_crd.x+2*_update_size.y*_update_size.z+_update_size.x*_update_size.z;
            case 4:
                return cur_buf_crd.y*_update_size.x+cur_buf_crd.x+2*_update_size.y*_update_size.z+2*_update_size.x*_update_size.z;
            case 5:
                return cur_buf_crd.y*_update_size.x+cur_buf_crd.x+2*_update_size.y*_update_size.z+2*_update_size.x*_update_size.z+_update_size.x*_update_size.y;
            default:
                printf("wrong face!\n");
                return -1;
        }
    }

    __device__
    int decide_bdr_face(const int3& cur_buf_crd)
    {
        if(cur_buf_crd.x ==0)
        {
            return 0;
        }else if (cur_buf_crd.x == _update_size.x-1) {
            return 1;
        }else if (cur_buf_crd.y == 0) {
            return 2;
        }else if (cur_buf_crd.y == _update_size.y-1) {
            return 3;
        }else if (cur_buf_crd.z == 0) {
            return 4;
        }else if (cur_buf_crd.z == _update_size.z-1) {
            return 5;
        }
        return -1;
    }

    void convertCostMap()
    {
        copy_ogm_2_host();
        copy_edt_2_host();
        for(int i=0;i<_map_volume; i++)
        {
            seendist_out[i].d = edt_H[i];
            seendist_out[i].o = glb_type_H[i];
        }
    }
    // for EDT
    __host__ __device__ __forceinline__
    int id(int x, int y, int z, int phase=0) const
    {
        switch (phase)
        {
            case 0:
                return z*_update_size.x*_update_size.y+y*_update_size.x+x;
            case 1:
                return z*_update_size.y*_update_size.x+y*_update_size.y+x;
            case 2:
                return z*_update_size.y*_update_size.z+y*_update_size.y+x;
            default:
                return z*_update_size.x*_update_size.y+y*_update_size.x+x;
        }
    }

    __device__
    float& edt_gpu_out(int x, int y, int z)
    {
        return _edt_D[id(x,y,z)];
    }

    __device__
    int& g_aux(int x, int y, int z, int phase)
    {
        return _aux[id(x,y,z,phase)];
    }

    __device__
    int& g(int x, int y, int z, int phase)
    {
        return _g[id(x,y,z,phase)];
    }

    __device__
    int& s(int x, int y, int z, int phase)
    {
        return _s[id(x,y,z,phase)];
    }

    __device__
    int& t(int x, int y, int z, int phase)
    {
        return _t[id(x,y,z,phase)];
    }

    //---
    __device__
    int3& coc(int x, int y, int z)
    {
        int idx_1d = id(x,y,z,0);
        if(idx_1d>_map_volume)
            assert(false);
        return _coc[idx_1d];
    }

    __device__
    int3& coc_aux(int x, int y, int z)
    {
        int idx_1d = id(x,y,z,0);
        if(idx_1d>_map_volume)
            assert(false);
        return _aux_coc[idx_1d];
    }

    __device__
    int f(int y,int i,int x,int z)
    {
        int a = g_aux(x,i,z,1);
        return (y-i)*(y-i) + a*a;

    }

    __device__
    int sep(int i,int u,int x,int z)
    {
        int a = g_aux(x,u,z,1);
        int b = g_aux(x,i,z,1);
        return (u*u-i*i+ a*a-b*b)/(2*(u-i));
    }

    __device__
    int f_z(int y,int i,int z,int x,int z_compress = 1)
    {
        return z_compress*(y-i)*(y-i) + g_aux(x,i,z,2);
    }

    __device__
    int sep_z(int i,int u,int z,int x,int z_compress = 1)
    {
        return (z_compress*u*u-z_compress*i*i+g_aux(x,u,z,2)-g_aux(x,i,z,2))/(2*z_compress*(u-i));
    }


    // var
    float _voxel_width;
    int3 _update_size;

    unsigned char _occu_thresh;
    float _update_max_h, _update_min_h;
    int3 _pvt;
    int _max_width;
    int _max_loc_dist_sq;
    int _bdr_num;
    int _map_volume;
    
    char *glb_type_H;
    float *edt_H;
    // for motion planner
    SeenDist *seendist_out;

    // ogm var
    int *_ray_count;
    char *_inst_type; // Visible in this particular scan; valid only in this scan
    char *_glb_type; //  copy global type to local. valid in the whole map cycle

    // EDT var
    float *_edt_D;
    int *_aux;
    int *_g;
    int *_s;
    int *_t;
    int3 *_coc;
    int3 *_aux_coc;

    // for motion planner
    int3 _half_shift;
    int seendist_size;
    float3 _msg_origin; // lower_left pos

    // for wave
    int _cutoff_grids_sq;
};

#endif //SRC_LOCAL_BATCH_H
