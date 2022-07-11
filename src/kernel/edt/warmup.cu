//
// Created by joseph on 22-7-7.
//

#include "warmup.h"

__global__ void warm_up_gpu() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid ;
}

void warmupCuda() {
    warm_up_gpu<<<64, 128>>>();
    cudaDeviceSynchronize();
}