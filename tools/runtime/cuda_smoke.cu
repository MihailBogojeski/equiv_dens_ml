// Minimal compiler/runtime check, independent of prebuilt Python CUDA wheels.
#include <cuda_runtime.h>
#include <cstdio>

__global__ void scale(double *values) {
    values[threadIdx.x] = 2.0 * threadIdx.x;
}

int main() {
    double *device = nullptr, host[32];
    if (cudaMalloc(&device, sizeof(host)) != cudaSuccess) return 1;
    scale<<<1, 32>>>(device);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 2;
    if (cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost) != cudaSuccess) return 3;
    for (int i = 0; i < 32; ++i) if (host[i] != 2.0 * i) return 4;
    if (cudaFree(device) != cudaSuccess) return 5;
    std::puts("nvcc compiled CUDA kernel and device results: passed");
    return 0;
}
