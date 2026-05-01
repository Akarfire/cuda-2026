#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16

__global__ void blockGEMM_kernel(const float* a, const float* b, float* c, int n) {

    __shared__ float b_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_B[BLOCK_SIZE][BLOCK_SIZE];
    
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    float sum = 0.0F;
    for (int m = 0; m < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {

        b_A[threadIdx.y][threadIdx.x] = a[row * n + (m * BLOCK_SIZE + threadIdx.x)];
        b_B[threadIdx.y][threadIdx.x] = b[(m * BLOCK_SIZE + threadIdx.y) * n + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) 
            sum += b_A[threadIdx.y][k] * b_B[k][threadIdx.x];

        __syncthreads();
    }
    if (row < n && col < n)
        c[row * n + col] = sum;
    
}


std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) 
{
    float* a_gpu;
    float* b_gpu;
    float* c_gpu;
    int bytes = n * n * sizeof(float);

    cudaMalloc(&a_gpu, bytes);
    cudaMalloc(&b_gpu, bytes);
    cudaMalloc(&c_gpu, bytes);

    cudaMemcpy(a_gpu, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
    
    blockGEMM_kernel<<<grid_size, block_size>>>(a_gpu, b_gpu, c_gpu, n);

    std::vector<float> result(n * n);

    cudaMemcpy(result.data(), c_gpu, bytes, cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
    return result;
}