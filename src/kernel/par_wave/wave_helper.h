
#ifndef SRC_WAVE_HELPER_CUH
#define SRC_WAVE_HELPER_CUH

#include "wave_core.cuh"
#include "frontier_wrapper.h"

template <class Mtype>
__forceinline__
void parWave(int3* frontier, int3* aux_frontier, int num_dirs, const int3 *dirs_D, int map_ct,
        Mtype *local_map, waveWrapper<int3> &wave, HASH_BASE* hash_base,
             bool display_glb_edt, int3* changed_keys, int* changed_cnt)
{
    using  namespace thrust;

    int num_t;//number of threads
    int num_of_blocks;
    int num_of_threads_per_block;
    int odd_even=0;

    do
    {
        num_t = wave.f_num_shared[0];
        wave.f_num_shared[0] =0;

        if(num_t>wave.max_nodes)
        {
            printf("Caution! Capacity is not enough\n");
            assert(false);
        }
        if(num_t==0)
            break;

        num_of_blocks = 1;
        if(num_t <NUM_BIN)
            num_of_threads_per_block = NUM_BIN;
        else if(num_t>MAX_THREADS_PER_BLOCK)
        {
            num_of_blocks = (int)ceil(num_t/(double)MAX_THREADS_PER_BLOCK);
            num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
        } else
        {
            num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
        }

        dim3 grids(num_of_blocks, 1, 1);
        dim3 threads(num_of_threads_per_block, 1, 1);

        int gray_shade;
        int3 * d_q1;
        int3 * d_q2;
        if(odd_even%2 ==0)
        {
            d_q1= raw_pointer_cast(&frontier[0]);
            d_q2= raw_pointer_cast(&wave.q2_shared[0]);
            gray_shade = GRAY0;
        }else
        {
            d_q2= raw_pointer_cast(&frontier[0]);
            d_q1= raw_pointer_cast(&wave.q2_shared[0]);
            gray_shade = GRAY1;
        }

        if(num_of_blocks ==1)
        {

            BFS_in_block<int3,Mtype><<< grids, threads >>>(wave, d_q1, d_q2, aux_frontier,
                    num_dirs, dirs_D, num_t, gray_shade,
                    *local_map, *hash_base, map_ct,
                    display_glb_edt, changed_keys, changed_cnt);

        }else
        {
            BFS_one_layer<int3,Mtype><<< grids, threads >>>(wave, d_q1, d_q2, aux_frontier,
                    num_dirs, dirs_D, num_t, gray_shade,
                    *local_map, *hash_base, map_ct,
                    display_glb_edt, changed_keys, changed_cnt);

        }
        odd_even++;

        int is_overflow = wave.overflow_shared[0];
        if(is_overflow)
        {
            printf("Error: local queue was overflow. Need to increase W_LOCAL_QUEUE\n");
            assert(false);
            break;
        }


    } while (1);
    wave.reset();
}
#endif //SRC_WAVE_HELPER_CUH
