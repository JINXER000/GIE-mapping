

#ifndef SRC_LOCKSET_H
#define SRC_LOCKSET_H

#include <cuda_runtime.h>
#include <assert.h>

/**
 * */

namespace vhashing { namespace detail {

        typedef int Lock;

        template <int N>
        struct LockSet {
            enum {
                CLS_FREE = 0,
                CLS_LOCKED = -1,
            };
            Lock *locks[N];

            __host__ __device__
            LockSet()
                    : locks() {
            }

            __host__ __device__
            ~LockSet() {
                YieldAll();
            }

            /**
           * add a lock to the list
             * */
            __device__ __host__
            bool TryLock(Lock &lock) {
#ifdef __CUDA_ARCH__
                int final_pos = -1;

		// insert into table
		for (int i=0; i<N; i++) {
			if (locks[i] == 0) {
				locks[i] = &lock;
				final_pos = i;
				break;
			}
		}

		// no space
    assert(final_pos != -1);

		// check if already acquired (write)
		for (int i=0; i<N; i++) {
			if (locks[i] == &lock && i != final_pos) {
        return true;
			}
		}

    // if not acquired, then we should switch it to LOCKED
		// try to write
		// the attempt invalidates ALL read locks -- because
		// a writer is already going to change the values
		int result = atomicCAS(&lock, (int)CLS_FREE, (int)CLS_LOCKED);

		if (result == CLS_LOCKED) { // failed to acquire
			// invalidate this lock
			locks[final_pos] = 0;
      return false;
		}
    else if (result == CLS_FREE) { // acquired
			return true;
		}
    else {
      assert(false);
      return false;
    }
#else
                // FIXME: implement c++11 <atomic> or something
                return true;
#endif
            }

            /**
             * Yield the specified lock
             *
             * First releases the write locks
             * then releases the read locks
             *
             * */
            __device__ __host__
            void Yield(Lock &lock) {
#ifdef __CUDA_ARCH__
                // check if already acquired (write)
		for (int i=0; i<N; i++) {
			if (locks[i] == &lock) {

				// check that we are not double-freeing
				for (int j=i+1; j<N; j++) {
					if (locks[j] == &lock) {
						locks[j] = 0;
						return;
					}
				}

				// no 2nd lock -- free this lock
				_YieldLock(lock);
				locks[i] = 0;
        return;
			}
		}
#endif
            }

#ifdef __CUDACC__
            __device__
	void _YieldLock(Lock &lock) {
    int result = atomicExch(&lock, (int)CLS_FREE);
    assert(result == (int)CLS_LOCKED);
	}
#endif

            /**
             * */
            __device__ __host__
            void YieldAll() {
#ifdef __CUDA_ARCH__
                // yield writers
		for (int i=0; i<N; i++) {
			if (locks[i]) {
				_YieldLock(*locks[i]);

				for (int j=i+1; j<N; j++) { // avoid double-free
					if (locks[i] == locks[j]) {
						locks[j] = 0; // clear additional write locks
					}
				}
				locks[i] = 0;
			}
		}
#endif
            }
        };

    }
}

#endif //SRC_LOCKSET_H
