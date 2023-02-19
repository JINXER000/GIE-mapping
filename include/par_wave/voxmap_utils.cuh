#ifndef VOXMAP_UTILS_CUH
#define VOXMAP_UTILS_CUH

#include <vox_hash/vhashing.h>
#include "map_structure/local_batch.h"

#define MAX_THREADS_PER_BLOCK 512
#define EMPTY_VALUE 999999
#define EMPTY_KEY  int3{EMPTY_VALUE,EMPTY_VALUE,EMPTY_VALUE}
#define VB_WIDTH 8
#define VB_SIZE 512

#define NUM_BIN 8 //the number of duplicated frontiers used in BFS_kernel_multi_blk_inGPU
#define EXP 3 // EXP = log(NUM_BIN), assuming NUM_BIN is still power of 2 in the future architecture
# define EXPSQ 6
#define MOD_OP 7 // This variable is also related with NUM_BIN; may change in the future architecture;
#define WHITE 16677217
#define GRAY0 16677219
#define GRAY1 16677220
#define GRAY2 16677221
#define GRAY3 16677222
#define BLACK 16677223
#define W_QUEUE_SIZE 400

#define RAISE_OUT 0
#define LOWER_OUT 1
#define LOWER_IN 3

struct   GlbVoxel {// 1+1+4+12 +4+4  +8 =34 bytes
    unsigned char occ_val = 0;
    char vox_type =  VOXTYPE_UNKNOWN;
    // time of being updated (mapping cycle)
    int update_ct = 0;

    // the closest obstacle coordinate in glb map
    int3 coc_glb = EMPTY_KEY;
    // squared distance
    int dist_sq = EMPTY_VALUE;
    // in one mapping cycle, the step indicator of propagation
    int wave_layer = -1;

    // for atomic
    Dist_id dist_id_pair;
};

struct VoxelBlock {
    GlbVoxel voxels[VB_SIZE];
};

struct CrdLessThan
{
    __device__ __host__
    bool operator()(int3 a, int3 b) {
        return (a.x < b.x) || (a.x == b.x && (
                (a.y < b.y) || (a.y == b.y && (
                        (a.z < b.z)))));
    }
};
struct CrdEqualTo
{
    __device__ __host__
    bool operator()(int3 a, int3 b) const{
        return (a.x == b.x) &&
               (a.y == b.y) &&
               (a.z == b.z);
    }
};

struct BlockHasher {
    __device__ __host__
    size_t operator()(int3 patch) const {
        const size_t p[] = {
                73856093,
                19349669,
                83492791
        };
        return ((size_t)patch.x * p[0]) ^
               ((size_t)patch.y * p[1]) ^
               ((size_t)patch.z * p[2]);
    }
};

typedef  vhashing::HashTable<int3, VoxelBlock, BlockHasher, CrdEqualTo, vhashing::device_memspace> D_HASH_TB;
typedef  vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, CrdEqualTo> HASH_BASE;


// useful helper functions


////!  get the key of the voxel block given a coord
////! \param hash_key
////! \return key of the voxel block
__host__ __device__ __forceinline__
int3 get_VB_key(const int3 &hash_key)
{
    int3 VB_key;
    VB_key.x=(hash_key.x >> EXP) - ((hash_key.x & MOD_OP) < 0);
    VB_key.y=(hash_key.y >> EXP )- ((hash_key.y & MOD_OP) < 0);
    VB_key.z=(hash_key.z >> EXP) - ((hash_key.z & MOD_OP) < 0);
    return VB_key;
}

__device__  __forceinline__
int get_voxID_in_VB(const int3 &glb_coord)
{
    return (((glb_coord.x & MOD_OP) +VB_WIDTH) & MOD_OP)*VB_WIDTH *VB_WIDTH
           + (((glb_coord.y & MOD_OP )+VB_WIDTH) & MOD_OP) *VB_WIDTH
           + (((glb_coord.z & MOD_OP) +VB_WIDTH) & MOD_OP);
}

__host__  __forceinline__
int3 reconstruct_vox_crd(const int3 &blk_offset, const int &idx_1d)
{
    int3 ijk;
    ijk.z = idx_1d & MOD_OP;
    ijk.y = ((idx_1d-ijk.z) >> EXP) & MOD_OP;
    ijk.x =  ((idx_1d - ijk.z - ijk.y*VB_WIDTH) >>EXPSQ) & MOD_OP;

    return make_int3(blk_offset.x + ijk.x, blk_offset.y + ijk.y, blk_offset.z + ijk.z);
}

//!  get the address of a voxel
//! \param crd
//! \param Vb
//! \return
__device__ __forceinline__
GlbVoxel* retrive_vox_D(const int3 &crd, VoxelBlock * Vb)
{
    int id_in_blk= get_voxID_in_VB(crd);
    GlbVoxel* rt_vox=&(Vb->voxels[id_in_blk]);
    return rt_vox;
}


__device__ __host__ __forceinline__
unsigned int get_squred_dist(const int3 &c1,const int3 &c2)
{
    int3 diff =c1-c2;
    unsigned int ret = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
    if(ret>900000 )
    {
        assert(false);
    }
    return ret;
}

__device__ __host__ __forceinline__
bool invalid_blk_key(const int3 &key)
{
    CrdEqualTo eq;
    return eq(key, EMPTY_KEY);
}

__device__ __host__ __forceinline__
bool invalid_src(const int3 &crd)
{
    CrdEqualTo eq;
    return eq(crd, EMPTY_KEY);
}

__device__ __host__ __forceinline__
bool invalid_dist_glb(const int &distsq)
{
    return distsq<0 || distsq >=900000;
}

__device__ __host__ __forceinline__
bool invalid_coc_glb(const int3 coc)
{
    bool ret=  ((coc.x>900000) || (coc.y > 900000) || (coc.z > 900000));
    return ret;
}

__device__ __host__ __forceinline__
bool invalid_coc_buf(const int3 &coc, const int loc_max_width)
{
    return  coc.x>loc_max_width || coc.y > loc_max_width || coc.z > loc_max_width ||
            coc.x <0 || coc.y<0 || coc.z<0;
}

__device__ __forceinline__
void set_hashvoxel_occ_val(GlbVoxel* glb_vox, float val, float low_pass_param, unsigned char occupancy_threshold, int time)
{

    // Calculate the occupancy probability using a low pass filter
    if (glb_vox->vox_type!=VOXTYPE_UNKNOWN)
        val = low_pass_param*val + (1.0f-low_pass_param)*static_cast<float>(glb_vox->occ_val);
    else
        val = low_pass_param*val + (1.0f-low_pass_param)*0.0f;

    if (val > UCHAR_MAX-1) val = UCHAR_MAX-1;
    if (val < 1) val = 1;

    // Assign the value back to the map
    glb_vox->occ_val = static_cast<unsigned char>(val);
    if(glb_vox->occ_val>occupancy_threshold)
        glb_vox->vox_type = VOXTYPE_OCCUPIED;
    else
        glb_vox->vox_type = VOXTYPE_FREE;
}

__device__ __forceinline__
bool insideAABB(float3 pt, float3 ll, float3 ur)
{
    bool inside =( pt.x>= ll.x && pt.y >=ll.y && pt.z >=ll.z) && (pt.x <= ur.x && pt.y <= ur.y && pt.z <= ur.z);
    return  inside;
}

#endif