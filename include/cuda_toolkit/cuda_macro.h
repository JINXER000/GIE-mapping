#ifndef CUDA_MACRO_H
#define CUDA_MACRO_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "matrix.cuh"
#include "se3.cuh"
#include "assert.h"

#define PNT_TYPE float3
#define REALSENSE_DEPTH_TPYE float
#define SCAN_DEPTH_TPYE float
#define LASER_RANGE_TPYE float
#define SENS_FAR_DIST 1000.f
#define GPU_PI_FLOAT 3.1415926f


#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at: %s : %d\n", file,line);
        fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);;
        exit(1);
    }
}

#define GPU_MALLOC(devPtr,size) checkCudaErrors(cudaMalloc(devPtr,size))
#define GPU_MEMCPY_H2D(dst,src,count) checkCudaErrors(cudaMemcpy(dst,src,count,cudaMemcpyHostToDevice))
#define GPU_MEMCPY_D2H(dst,src,count) checkCudaErrors(cudaMemcpy(dst,src,count,cudaMemcpyDeviceToHost))
#define GPU_MEMCPY_D2D(dst,src,count) checkCudaErrors(cudaMemcpy(dst,src,count,cudaMemcpyDeviceToDevice))
#define GPU_FREE(devPtr) checkCudaErrors(cudaFree(devPtr))
#define GPU_MEMSET(devPtr,value,count) checkCudaErrors(cudaMemset(devPtr,value,count))
#define GPU_DEV_SYNC() checkCudaErrors(cudaDeviceSynchronize())



#endif // CUDA_MACRO_H
