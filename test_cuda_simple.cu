#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_kernel() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    printf("Starting CUDA test...\n");
    
    // 检查CUDA设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA devices\n", deviceCount);
    
    if (deviceCount > 0) {
        test_kernel<<<1, 5>>>();
        cudaDeviceSynchronize();
        printf("CUDA test completed successfully\n");
    } else {
        printf("No CUDA devices found\n");
    }
    
    return 0;
}