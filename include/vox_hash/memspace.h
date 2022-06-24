

#ifndef SRC_MEMSPACE_H
#define SRC_MEMSPACE_H


#include "cuda_toolkit/cuda_macro.h"
#include <thrust/host_vector.h>

namespace vhashing {

    typedef thrust::device_system_tag device_memspace;
    typedef thrust::host_system_tag host_memspace;

    template <typename T, typename M>
    struct ptr_type {};

    template <typename T>
    struct ptr_type<T, host_memspace> {
        typedef T* type;
    };
    template <typename T>
    struct ptr_type<T, device_memspace> {
        typedef thrust::device_ptr<T> type;
    };

    template <typename T, typename M>
    struct vector_type {};

    template <typename T>
    struct vector_type<T, host_memspace> {
        typedef thrust::host_vector<T> type;
    };
    template <typename T>
    struct vector_type<T, device_memspace> {
        typedef thrust::device_vector<T> type;
    };


    namespace detail {

        template <typename T>
        T* memspace_alloc(size_t num_elems, host_memspace) {
            T* t = malloc(num_elems * sizeof(T));
            assert(t);
            return t;
        }

        template <typename T>
        T* memspace_alloc(size_t num_elems, device_memspace) {
            T* t = 0;
//            cudaMalloc(&t, num_elems * sizeof(T));
            GPU_MALLOC(&t, num_elems * sizeof(T));
            return t;
        }

        template <typename T>
        void memspace_fill(T* start, T* end, const T &t, host_memspace) {
            thrust::uninitialized_fill(start, end, t);
        }

        template <typename T>
        void memspace_fill(T* start, T* end, const T &t, device_memspace) {
            thrust::uninitialized_fill(
                    thrust::device_pointer_cast(start),
                    thrust::device_pointer_cast(end),
                    t);
        }


        template <typename memspace>
        struct memspace_deleter {
            template <typename T>
            void operator() (const T *t) {
                thrust::free(memspace(), (void*)t);
            }
        };

    }  // namespace detail
}  // namespace vhashing

#endif //SRC_MEMSPACE_H
