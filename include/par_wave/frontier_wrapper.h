
#ifndef SRC_FRONTIER_WRAPPER_H
#define SRC_FRONTIER_WRAPPER_H

#include <iostream>
#include "vox_hash/memspace.h"

template <class Ktype>
struct waveBase
{
    int *f_num;
    int *switchK;
    int *overflow;
    int *wave_type;
    int *aux_num;

    Ktype *q2D;
    Ktype *q_aux;

    waveBase()
    {

    }
};

template <class Ktype, class memspace=vhashing::device_memspace>
class waveWrapper: public waveBase<Ktype>
{
public:

    int max_nodes;

    typename vhashing::vector_type<Ktype,memspace>::type q2_shared;
    typename vhashing::vector_type<int,memspace>::type f_num_shared;
    typename vhashing::vector_type<int,memspace>::type switchk_shared;
    typename vhashing::vector_type<int,memspace>::type overflow_shared;
    typename vhashing::vector_type<int,memspace>::type wave_type_shared;
    typename vhashing::vector_type<int,memspace>::type aux_num_shared;

    waveWrapper(int nv, int src_type):waveBase<Ktype>(), max_nodes(nv),
            q2_shared(nv),f_num_shared(1), switchk_shared(1), overflow_shared(1),
            wave_type_shared(1),aux_num_shared(1)
    {
        wave_type_shared[0] = src_type;
        reset();
        using thrust::raw_pointer_cast;
        this->q2D = raw_pointer_cast(&q2_shared[0]);
        this->f_num = raw_pointer_cast(&f_num_shared[0]);
        this->switchK = raw_pointer_cast(&switchk_shared[0]);
        this->overflow = raw_pointer_cast(&overflow_shared[0]);
        this->wave_type = raw_pointer_cast(&wave_type_shared[0]);
        this->aux_num = raw_pointer_cast(&aux_num_shared[0]);

        thrust::fill(q2_shared.begin(),q2_shared.end(),EMPTY_KEY);

    }

    void reset()
    {
        f_num_shared[0]= 1;
        overflow_shared[0]= 0;
    }
};
#endif //SRC_FRONTIER_WRAPPER_H
