#include <stdio.h>

__global__ void dummy_kernel() { }

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
    }
    
    dummy_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
