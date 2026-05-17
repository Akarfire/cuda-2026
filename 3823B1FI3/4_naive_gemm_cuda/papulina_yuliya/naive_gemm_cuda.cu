#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <memory>

struct Deleter {
    void operator()(float* ptr) const {
        if (ptr) cudaFree(ptr);
    }
};
using my_pointer = std::unique_ptr<float[], Deleter>;

template <int SIZE>
__global__ void GemmKernel(const float* __restrict__ A, 
                                const float* __restrict__ B, 
                                float* __restrict__ C, 
                                int n) {
    __shared__ float shared_A[SIZE][SIZE];
    __shared__ float shared_B[SIZE][SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = blockIdx.y * SIZE + ty;
    int j = blockIdx.x * SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (n + SIZE - 1) / SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        if (i < n && (t * SIZE + tx) < n)
            shared_A[ty][tx] = A[i * n + t * SIZE + tx];
        else
            shared_A[ty][tx] = 0.0f;

        if ((t * SIZE + ty) < n && j < n)
            shared_B[ty][tx] = B[(t * SIZE + ty) * n + j];
        else
            shared_B[ty][tx] = 0.0f;

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < SIZE; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }
        __syncthreads();
    }

    if (i < n && j < n) {
        C[i * n + j] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> result(n * n);
    size_t byte_size = static_cast<size_t>(n) * n * sizeof(float);

    float *A = nullptr, *B = nullptr, *C = nullptr;
    
    cudaMalloc(&A, byte_size);
    my_pointer d_A(A);
    
    cudaMalloc(&B, byte_size);
    my_pointer d_B(B);
    
    cudaMalloc(&C, byte_size);
    my_pointer d_C(C);

    cudaMemcpy(d_A.get(), a.data(), byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.get(), b.data(), byte_size, cudaMemcpyHostToDevice);
    constexpr int dim = 32;
    dim3 block_dim(dim, dim);
    dim3 grid_dim((n + dim - 1) / dim, (n + dim - 1) / dim);

    GemmKernel<dim><<<grid_dim, block_dim>>>(d_A.get(), d_B.get(), d_C.get(), n);
    cudaDeviceSynchronize();
    cudaMemcpy(result.data(), d_C.get(), byte_size, cudaMemcpyDeviceToHost);

    return result;
}