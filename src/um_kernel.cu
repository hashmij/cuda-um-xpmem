#include <stdio.h>
#include "kernel.h"

__global__ void memset_kernel(char *buf, size_t n, char c) {
    for (int i=0; i<n; i++) {
        buf[i] = c;
    }
    __syncthreads();
}

__global__ void print_kernel(char *buf, size_t n) {
    for (int i=0; i<n; i++) {
        printf("%c", buf[i]);
    }
    printf("\n");
    __syncthreads();
}

__global__ void memset_kernel_mt(char *sbuf, size_t n, char c) {

    /* multiple threads loop over the data and set the values */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 
    int i;
    for (i = tid; i < n; i+=stride) {
        if (tid < n) {
            sbuf[i] = c;
        }
    }
}

extern "C" void do_memset_device_mt(char *buf, size_t n, char c) {
    memset_kernel<<< 32, 256>>>(buf, n, c);
    cudaDeviceSynchronize();
}

extern "C" void do_memset_device(char *buf, size_t n, char c) {
    memset_kernel<<< 1, 1 >>>(buf, n, c);
    cudaDeviceSynchronize();
}

extern "C" void do_print_gpu(char *buf, size_t n) {
    print_kernel<<< 1, 1 >>>(buf, n);
    cudaDeviceSynchronize();
}
