

#include "unify_helper.cuh"
#include "alloc_helper.cuh"
#include "wave_helper.h"
#include "vox_hash/memspace.h"
#include "par_wave/glb_hash_map.h"

GlbHashMap::GlbHashMap(int bdr_size, int3 loc_dim, int bucket_max, int block_max)
{
    int loc_volume_size = loc_dim.x*loc_dim.y*loc_dim.z;
    int max_keys_in_local = (loc_dim.x/VB_WIDTH+1)* (loc_dim.y/VB_WIDTH+1)*(loc_dim.z/VB_WIDTH+1);
    require_alloc_keys_D.resize(max_keys_in_local);
    success_alloc_D.resize(max_keys_in_local+1);

    VB_values_H.resize(block_max);
    VB_keys_H.resize(block_max);
    std::fill(VB_keys_H.begin(),VB_keys_H.end(),EMPTY_KEY);
    VB_cnt_H = 0;

    VB_keys_loc_D.resize(loc_volume_size);

    stream_VB_vals_D.resize(loc_volume_size);
    stream_VB_keys_D.resize(loc_volume_size);
    thrust::fill(stream_VB_keys_D.begin(),stream_VB_keys_D.end(),EMPTY_KEY);
    changed_cnt.resize(1);
    changed_cnt[0] = 0;

    frontierA.resize(bdr_size);
    frontierB.resize(bdr_size);
    frontierC.resize(bdr_size);
    thrust::fill(frontierA.begin(),frontierA.end(),EMPTY_KEY);
    thrust::fill(frontierB.begin(),frontierB.end(),EMPTY_KEY);
    thrust::fill(frontierC.begin(),frontierC.end(),EMPTY_KEY);

    waveA = new waveWrapper<int3>(bdr_size, RAISE_OUT);
    waveB = new waveWrapper<int3>(bdr_size, LOWER_OUT);
    waveC = new waveWrapper<int3>(bdr_size, LOWER_IN);

    // cannot init at the beginning
    hash_table_D= new D_HASH_TB(bucket_max, 2, block_max, EMPTY_KEY);

    GPU_MALLOC(&dirs_6_D, num_dirs_6*sizeof(int3));
    GPU_MEMCPY_H2D(dirs_6_D, dirs_6_H, num_dirs_6*sizeof(int3));

    std::cout<<"Hash Table Initialized"<<std::endl;
}

GlbHashMap::~GlbHashMap()
{
    GPU_FREE(dirs_6_D);
}

void GlbHashMap::setLocMap(LocMap *lMap) {
    _lMap = lMap;
}

void GlbHashMap::allocHashTB()
{
    using namespace thrust;
    using namespace vhashing;

    thrust::sort(VB_keys_loc_D.begin(),
                 VB_keys_loc_D.end(),
                 CrdLessThan());
    auto last_it_unique = thrust::unique(VB_keys_loc_D.begin(),
                                  VB_keys_loc_D.end(),CrdEqualTo());

    int num_elems = thrust::distance(VB_keys_loc_D.begin(), last_it_unique);
    bool all_successful = true;

    do {
        // unique key that not been allocated
        auto last_it_require = thrust::copy_if(VB_keys_loc_D.begin(), last_it_unique,
                                       require_alloc_keys_D.begin(),
                                       RequiresAllocation<HASH_BASE>{*(hash_table_D)});

        /* request the empty blocks */
        int numJobs = thrust::distance(require_alloc_keys_D.begin(), last_it_require);// new blk need to be allocated

        int numBlocks = (numJobs + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
        // copy all keys only at the first time

        thrust::device_ptr<int> d_link_head((hash_table_D)->alloc.link_head);
        int h_link_head = *d_link_head;
        printf("New blocks allocated: %d, unused blks: %d\n", numJobs, h_link_head);
        if (numJobs == 0)
            return;

        int blk_start = (hash_table_D)->alloc.allocate_n(numJobs);// 19995

        device_ptr<int> unsuccessful(&success_alloc_D[numJobs]);// last elem of success


        *unsuccessful = 0;
        TryAllocateKernel<<<numBlocks, MAX_THREADS_PER_BLOCK>>>
                (*(hash_table_D),
                 raw_pointer_cast(&require_alloc_keys_D[0]),
                 raw_pointer_cast(&success_alloc_D[0]),
                 blk_start,
                 numJobs);


        ReturnAllocations<<<numBlocks, MAX_THREADS_PER_BLOCK>>>
                (*(hash_table_D),
                 raw_pointer_cast(&success_alloc_D[0]),
                 raw_pointer_cast(&unsuccessful[0]), numJobs);


        all_successful = !((int)*unsuccessful);
    } while (!all_successful);

}

void GlbHashMap::updateHashOGM(bool input_pyntcld, const int map_ct, bool stream_glb_ogm)
{

    allocHashTB();
    const int gridSize = _lMap->_local_size.z;
    const int blkSize = _lMap->_local_size.y;
    if(input_pyntcld)
    {
        updateHashOGMWithPntCld<<<gridSize,blkSize>>>(*_lMap, *hash_table_D, map_ct,stream_glb_ogm,
                                                      raw_pointer_cast(&stream_VB_keys_D[0]),
                                                      raw_pointer_cast(&changed_cnt[0]));
    }else
    {
        updateHashOGMWithSensor<<<gridSize,blkSize>>>(*_lMap, *hash_table_D, map_ct,stream_glb_ogm,
                                                      raw_pointer_cast(&stream_VB_keys_D[0]),
                                                      raw_pointer_cast(&changed_cnt[0]));

    }

}


void GlbHashMap::mergeNewObsv(const int map_ct, const bool display_glb_edt)
{

    MarkLimitedObserve<<<_lMap->_local_size.z, _lMap->_local_size.x>>>(*_lMap, *hash_table_D, display_glb_edt,
                                                                                  raw_pointer_cast(&stream_VB_keys_D[0]),
                                                                                  raw_pointer_cast(&changed_cnt[0]));

    waveA->f_num_shared[0] = 0;
    waveB->f_num_shared[0] = 0;
    waveC->f_num_shared[0] = 0;
    obtainFrontiers<<<_lMap->_local_size.z, _lMap->_local_size.x>>>(*_lMap, *hash_table_D,
                                                                                  raw_pointer_cast(&frontierA[0]),
                                                                                  raw_pointer_cast(&frontierB[0]),
                                                                                  raw_pointer_cast(&frontierC[0]),
                                                                                    raw_pointer_cast(&(waveA->f_num_shared[0])),
                                                                                    raw_pointer_cast(&(waveB->f_num_shared[0])),
                                                                                  raw_pointer_cast(&(waveC->f_num_shared[0])),
                                                                                  num_dirs_6, dirs_6_D,
                                                                                  map_ct);


    fA_num_raw = waveA->f_num_shared[0];
    fB_num_raw = waveB->f_num_shared[0];
    fC_num_raw = waveC->f_num_shared[0];


    waveA->f_num_shared[0] = fA_num_raw;

    if(!_lMap->_fast_mode)
    {
        waveA->aux_num_shared[0] = fB_num_raw;
        parWave<LocMap>(raw_pointer_cast(&frontierA[0]), raw_pointer_cast(&frontierB[0]),
                        num_dirs_6, dirs_6_D, map_ct,
                        _lMap, *waveA, hash_table_D,
                        display_glb_edt, raw_pointer_cast(&stream_VB_keys_D[0]), raw_pointer_cast(&changed_cnt[0]));
        thrust::fill(frontierA.begin(),frontierA.end(),EMPTY_KEY);
        fB_num_raw = waveA->aux_num_shared[0];

        waveB->f_num_shared[0] = fB_num_raw;
        waveB->aux_num_shared[0] = fC_num_raw;
        parWave<LocMap>(raw_pointer_cast(&frontierB[0]), raw_pointer_cast(&frontierC[0]),
                        num_dirs_6, dirs_6_D, map_ct,
                        _lMap, *waveB, hash_table_D,
                        display_glb_edt, raw_pointer_cast(&stream_VB_keys_D[0]), raw_pointer_cast(&changed_cnt[0]));
        thrust::fill(frontierB.begin(),frontierB.end(),EMPTY_KEY);
        fC_num_raw = waveB->aux_num_shared[0];
    }



    waveC->f_num_shared[0] = fC_num_raw;
    parWave<LocMap>(raw_pointer_cast(&frontierC[0]), raw_pointer_cast(&frontierA[0]),
                    num_dirs_6, dirs_6_D, map_ct,
                    _lMap, *waveC, hash_table_D,
                    display_glb_edt, raw_pointer_cast(&stream_VB_keys_D[0]), raw_pointer_cast(&changed_cnt[0]));
    thrust::fill(frontierC.begin(),frontierC.end(),EMPTY_KEY);

    UpdateHashBatch<<<_lMap->_local_size.z, _lMap->_local_size.x>>>(*_lMap, *hash_table_D, display_glb_edt,
                                                                       raw_pointer_cast(&stream_VB_keys_D[0]),
                                                                      raw_pointer_cast(&changed_cnt[0]));

}

void GlbHashMap::streamD2H(int changed_cnt_condense)
{
    for(int i=0; i<changed_cnt_condense; i++)
    {
        VoxelBlock* VB_addr = stream_VB_vals_D[i];
        int3 cg_key = stream_VB_keys_D[i];
        if(invalid_blk_key(cg_key))
        {
            continue;
        }
        auto tmp = hash_table_H_std.find(cg_key);
        if(tmp==hash_table_H_std.end())
        {
            hash_table_H_std.insert(std::make_pair(cg_key, VB_cnt_H));
            GPU_MEMCPY_D2H(&(VB_values_H[VB_cnt_H]).voxels,VB_addr->voxels, sizeof(GlbVoxel)*VB_SIZE);
            VB_keys_H[VB_cnt_H] = cg_key;
            VB_cnt_H++;
        }else
        {
            int VB_idx = tmp->second;
            GPU_MEMCPY_D2H(&(VB_values_H[VB_idx]).voxels,VB_addr->voxels, sizeof(GlbVoxel)*VB_SIZE);
        }
    }
}

void GlbHashMap::streamPipeline()
{
    int change_cnt_H = changed_cnt[0];
    thrust::sort(stream_VB_keys_D.begin(), stream_VB_keys_D.begin()+change_cnt_H, CrdLessThan());
    auto addr_end = thrust::unique(stream_VB_keys_D.begin(),stream_VB_keys_D.begin()+change_cnt_H,CrdEqualTo());
    int changed_cnt_condense = thrust::distance(stream_VB_keys_D.begin(), addr_end);
    const int grid_size = 1+(int)(changed_cnt_condense/MAX_THREADS_PER_BLOCK);
    getUpdatedAddr<<<grid_size,MAX_THREADS_PER_BLOCK>>>(raw_pointer_cast(&stream_VB_keys_D[0]),
                                                        raw_pointer_cast(&stream_VB_vals_D[0]),
                                                        *hash_table_D, changed_cnt_condense);

    streamD2H(changed_cnt_condense);
    changed_cnt[0] = 0;
}