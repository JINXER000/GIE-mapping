#ifndef MAP_UNIFY_CUH
#define MAP_UNIFY_CUH


#include <unordered_map>
#include "voxmap_utils.cuh"
#include "frontier_wrapper.h"
#include "map_structure/local_batch.h"
#include "map_structure/pre_map.h"

struct GlbHashMap {
public:
    GlbHashMap(int bdr_size, int3 loc_dim, int bucket_max, int block_max);
    ~GlbHashMap();

    void setLocMap(LocMap *lMap);

    void allocHashTB();

    void updateHashOGM(bool input_pynt, const int map_ct, bool stream_glb_ogm,
                       Ext_Obs_Wrapper* ext_obsv);

    void streamD2H(int changed_cnt_condense);

    void streamPipeline();

    void mergeNewObsv(const int map_ct, const bool display_glb_edt);

public:
    D_HASH_TB *hash_table_D;

    // key and value in host
    std::vector<VoxelBlock> VB_values_H;
    std::vector<int3> VB_keys_H;

    // hashtable in host
    std::unordered_map<int3, int,BlockHasher,CrdEqualTo> hash_table_H_std;
    int VB_cnt_H;

    // keys in local EDT
    thrust::device_vector<int3> VB_keys_loc_D;

    // for allocation
    thrust::device_vector<int3> require_alloc_keys_D;
    thrust::device_vector<int> success_alloc_D;

    // for GPU-CPU stream
    thrust::device_vector<VoxelBlock*> stream_VB_vals_D;
    thrust::device_vector<int3> stream_VB_keys_D;
    thrust::device_vector<int> changed_cnt;

    // for wavefront
    thrust::device_vector<int3> frontierA, frontierB, frontierC;
    int fA_num_raw, fB_num_raw, fC_num_raw;

    waveWrapper<int3> *waveA, *waveB, *waveC;

    // for connectivity
    const static int num_dirs_6 =6;// only faces
    const int3 dirs_6_H[num_dirs_6]={int3{-1, 0, 0},int3{1, 0, 0},int3{0, -1, 0},
                                   int3{0, 1, 0},int3{0, 0, -1},int3{0, 0, 1}};
    int3* dirs_6_D;

    LocMap *_lMap;
};


#endif