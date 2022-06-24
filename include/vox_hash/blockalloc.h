
#ifndef SRC_BLOCKALLOC_H
#define SRC_BLOCKALLOC_H

#include <limits>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_new.h>
#include <thrust/device_malloc.h>
#include <thrust/device_delete.h>
#include <thrust/sequence.h>


namespace vhashing {
    namespace detail {

        template <typename Value>
        struct BlockAllocBase {
            typedef int32_t ptr_t;

            static const int FREE=0;
            static const int LOCKED=1;

            int num_elems;
            Value     *data;
            int32_t  *offsets;
            int *mutex, *link_head;
            int h_link_head;

            BlockAllocBase(int num_elems = 0)
                    : num_elems(num_elems) {}

            __device__ __host__
            inline Value &operator[](int32_t pointer) const {
//                assert (pointer > 0 && pointer <= num_elems);
                if (pointer<=0 || pointer>num_elems)
                {
                    assert(false);
                }

                return data[pointer];
            }

            /** For bulk allocations to reduce contention on
             * link_head
             * */
            __host__
                    int32_t allocate_n(int n) {
                thrust::device_ptr<int> d_link_head(link_head);
                int tmp_link_head = *d_link_head;

                // subtract link head by n
                tmp_link_head -= n;
                if (tmp_link_head < 0) {
                    throw "out of block memory";
                }
//                if (tmp_link_head < 0) {
//                    tmp_link_head = h_link_head;
//                }
                *d_link_head = tmp_link_head;


                h_link_head = tmp_link_head;
                return h_link_head + 1;
            }

            /**
             * WARNING: all threads must allocate simultaneously. No free() allowed
             * in between, otherwise you will get data corruption */
            __device__ __host__
            ptr_t allocate() {
                int which;
#ifdef __CUDA_ARCH__
                which = atomicSub(link_head, 1);
#else
                which = (*link_head)--;
#endif

                ptr_t rv;
                if (which < 0) {
                    rv = 0;
                    assert(false);
                }
                else {
                    rv = offsets[which];
                }

#ifdef __CUDA_ARCH__
                // just so that any free() will still return counter to 0
		atomicMax(link_head, -1);
		__threadfence();
#endif

                return rv;
            }

            /**
             * WARNING: all threads must allocate simultaneously. No free() allowed
             * in between, otherwise you will get data corruption */
            __device__ __host__
            void free(ptr_t p) {
                if (p == 0) {
                    return;
                }
#ifdef __CUDA_ARCH__
                int which = atomicAdd(link_head, 1);
#else
                int which = (*link_head)++;
#endif

                offsets[which + 1] = p;


            }

        };

    }}
#endif //SRC_BLOCKALLOC_H
