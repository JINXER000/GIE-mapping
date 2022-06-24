
#ifndef SRC_LOCAL_BATCH_H
#define SRC_LOCAL_BATCH_H

#include <cuda_toolkit/cuda_macro.h>

#define VOXTYPE_OCCUPIED 2
#define VOXTYPE_FREE 1
#define VOXTYPE_UNKNOWN 0
#define VOXTYPE_FNT 3

#define XSHIFT 0
#define YSHIFT 11
#define ZSHIFT 22
#define XMASK 0x7ff
#define YMASK 0x7ff
#define ZMASK 0x3ff

struct SeenDist
{
    float d; // distance
    bool s;  // seen
    bool o; // used for thining algorithm
};

typedef union  {
    int sq_dist[2];                 // sq_dist[0] = lowest
    int parent_loc_id[2];                     // parent_loc_id[1] = lowIdx
    unsigned long long int ulong;    // for atomic update
} Dist_id;

class LocMap
{
public:
    LocMap(const float voxel_size, const int3 local_size,const unsigned char occupancy_threshold,
           const float ogm_min_h, const float ogm_max_h, const int cutoff_grids_sq):
            _voxel_width(voxel_size),
            _local_size(local_size),
            _occu_thresh(occupancy_threshold),
            _update_max_h(ogm_max_h),
            _update_min_h(ogm_min_h),
            _cutoff_grids_sq(cutoff_grids_sq)
    {
        _map_volume = local_size.x * local_size.y * local_size.z;
        _max_width = local_size.x + local_size.y + local_size.z;
        _max_loc_dist_sq = local_size.x*local_size.x + local_size.y*local_size.y + local_size.z*local_size.z;
        _bdr_num =  2*(local_size.x * local_size.y + local_size.y* local_size.z + local_size.x*local_size.z);
        _half_shift = make_int3(_local_size.x/2, _local_size.y/2, _local_size.z/2);
        seendist_size = _map_volume*sizeof(SeenDist);
        _wave_range.x = (((0xffffffff) >> XSHIFT) & XMASK)-1; // even numbers
        _wave_range.y = (((0xffffffff) >> YSHIFT) & YMASK)-1;
        _wave_range.z = (((0xffffffff) >> ZSHIFT) & ZMASK)-1;
        printf("Local Map initialized\n");
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

        GPU_MALLOC(&_loc_wave_layer, sizeof(int)*_map_volume);

        GPU_MALLOC(&_dist_id_pair, sizeof(Dist_id)*_map_volume);

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
        GPU_FREE(_loc_wave_layer);
        GPU_FREE(_dist_id_pair);

        delete [] glb_type_H;
        delete [] edt_H;
        delete [] seendist_out;
    }


    __device__
    bool is_inside_local_volume(const int3 & crd) const
    {
        if (crd.x<0 || crd.x>=_local_size.x ||
            crd.y<0 || crd.y>=_local_size.y ||
            crd.z<0 || crd.z>=_local_size.z)
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
        pivot.x -= _local_size.x/2;
        pivot.y -= _local_size.y/2;
        pivot.z -= _local_size.z/2;

        float3 origin = coord2pos(pivot);

        // set pvt
        _pvt = pivot;
        _msg_origin = origin;
        return pivot;
    }

    __device__ __host__
    bool is_inside_wave_range(const int3 & crd) const
    {
        if (crd.x<0 || crd.x>=_wave_range.x ||
            crd.y<0 || crd.y>=_wave_range.y ||
            crd.z<0 || crd.z>=_wave_range.z)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    __host__
    void calculate_update_pivot(float3 map_center)
    {
        _update_pvt = pos2coord(map_center);
        _update_pvt.x -= _wave_range.x/2;
        _update_pvt.y -= _wave_range.y/2;
        _update_pvt.z -= _wave_range.z/2;
    }
    __device__ __host__
    int coord2idx_local(const int3 &c) const
    {
        return c.x + c.y*_local_size.x + c.z*_local_size.x*_local_size.y;
    }

    __device__ __host__
    int3 id2wr_coc(const int &idx_1d)
    {
        int3 coc_in_update_range;
        coc_in_update_range.x = (((idx_1d) >> XSHIFT) & XMASK);
        coc_in_update_range.y = (((idx_1d) >> YSHIFT) & YMASK);
        coc_in_update_range.z = (((idx_1d) >> ZSHIFT) & ZMASK);

        return coc_in_update_range;
    }

    __device__ __host__
    int3 id2coc_glb(const int &idx_1d)
    {
        return  wave_range2glb(id2wr_coc(idx_1d));
    }

    __device__ __host__
    int3 id2coc_buf(const int &idx_1d)
    {
        return wave_range2loc(id2wr_coc(idx_1d));
    }

    __device__ __host__
    int wr_coc2idx(const int3 &coc_in_wr)
    {
        int idx_1d;
        if(!is_inside_wave_range(coc_in_wr))
        {
            printf("debug here: coc not in wave range\n");
            assert(false);
            return -1;
        }
        idx_1d = (((coc_in_wr.x) << XSHIFT) | ((coc_in_wr.y) << YSHIFT) | ((coc_in_wr.z) << ZSHIFT));
        return  idx_1d;
    }

    __device__ __host__
    int coc_glb2id(const int3 &coc_glb)
    {
        return  wr_coc2idx(glb2wave_range(coc_glb));
    }

    __device__ __host__
    int coc_buf2id(const int3 &coc_buf)
    {
        return wr_coc2idx(loc2wave_range(coc_buf));
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
    __host__ __device__
    int3 glb2wave_range(const int3 & c) const
    {
        return c-_update_pvt;
    }

    __host__ __device__
    int3 wave_range2glb(const int3 & c) const
    {
        return c+_update_pvt;
    }

    __host__ __device__
    int3 wave_range2loc(const int3 & c) const
    {
        return glb2loc(wave_range2glb(c));
    }

    __host__ __device__
    int3 loc2wave_range(const int3 & c) const
    {
        return glb2wave_range(loc2glb(c));
    }
    __device__
    int atom_add_type_count(const int3 &crd, int val)
    {
        if (!is_inside_local_volume(crd))
            return -1;

#ifdef __CUDA_ARCH__
        return atomicAdd(&(_ray_count[coord2idx_local(crd)]), val);
#endif
    }
    __device__
    int get_vox_count(const int3 &crd)
    {
        if (!is_inside_local_volume(crd))
            return 0;

        int idx = coord2idx_local(crd);
        return _ray_count[idx];
    }

    __device__
    void set_vox_count(const int3 &crd, int val)
    {
        if (!is_inside_local_volume(crd))
            return;

        int idx = coord2idx_local(crd);
        _ray_count[idx]= val;
    }
    __device__
    char get_vox_type(const int3 &crd)
    {
        if  (!is_inside_local_volume(crd))
            return VOXTYPE_UNKNOWN;

        int idx = coord2idx_local(crd);
        return _inst_type[idx];
    }

    __device__
    void set_vox_type(const int3 &crd, const char type)
    {
        if  (!is_inside_local_volume(crd))
            return;

        int idx = coord2idx_local(crd);
        _inst_type[idx] = type;
    }
    __device__
    char get_vox_glb_type(const int3 &crd)
    {
        if  (!is_inside_local_volume(crd))
            return VOXTYPE_UNKNOWN;

        int idx = coord2idx_local(crd);
        return _glb_type[idx];
    }

    __device__
    void set_vox_glb_type(const int3 &crd, const char type)
    {
        if  (!is_inside_local_volume(crd))
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
                return cur_buf_crd.z*_local_size.y+cur_buf_crd.y;
            case 1:
                return cur_buf_crd.z*_local_size.y+cur_buf_crd.y+_local_size.y*_local_size.z;
            case 2:
                return cur_buf_crd.z*_local_size.x+cur_buf_crd.x+2*_local_size.y*_local_size.z;
            case 3:
                return cur_buf_crd.z*_local_size.x+cur_buf_crd.x+2*_local_size.y*_local_size.z+_local_size.x*_local_size.z;
            case 4:
                return cur_buf_crd.y*_local_size.x+cur_buf_crd.x+2*_local_size.y*_local_size.z+2*_local_size.x*_local_size.z;
            case 5:
                return cur_buf_crd.y*_local_size.x+cur_buf_crd.x+2*_local_size.y*_local_size.z+2*_local_size.x*_local_size.z+_local_size.x*_local_size.y;
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
        }else if (cur_buf_crd.x == _local_size.x-1) {
            return 1;
        }else if (cur_buf_crd.y == 0) {
            return 2;
        }else if (cur_buf_crd.y == _local_size.y-1) {
            return 3;
        }else if (cur_buf_crd.z == 0) {
            return 4;
        }else if (cur_buf_crd.z == _local_size.z-1) {
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
                return z*_local_size.x*_local_size.y+y*_local_size.x+x;
            case 1:
                return z*_local_size.y*_local_size.x+y*_local_size.y+x;
            case 2:
                return z*_local_size.y*_local_size.z+y*_local_size.y+x;
            default:
                return z*_local_size.x*_local_size.y+y*_local_size.x+x;
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
    int& dist_in_pair(int idx_1d)
    {
        return _dist_id_pair[idx_1d].sq_dist[0];
    }

    __device__
    int& parent_id_in_pair(int idx_1d)
    {
        return _dist_id_pair[idx_1d].parent_loc_id[1];
    }

    __device__
    unsigned  long long int& ulong_in_pair(int idx_1d)
    {
        return _dist_id_pair[idx_1d].ulong;
    }

    __device__
    int& wave_layer(int x, int y, int z)
    {
        int idx_1d = id(x,y,z,0);
        return wave_layer(idx_1d);
    }
    __device__
    int& wave_layer(int idx_1d)
    {
        if(idx_1d>_map_volume)
            assert(false);
        return _loc_wave_layer[idx_1d];
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
    int3 _local_size;

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
    int * _loc_wave_layer;
    int _cutoff_grids_sq;
    Dist_id *_dist_id_pair;

    int3 _wave_range;
    int3 _update_pvt;
};

#endif //SRC_LOCAL_BATCH_H
