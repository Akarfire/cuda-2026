#include "gemm_cublas.h"
#include <memory>
#include <cuda_runtime.h>
#include <cublas_v2.h>

struct Deleter {
    void operator()(float* ptr) const {
        if (ptr) cudaFree(ptr);
    }
};
using my_pointer = std::unique_ptr<float[], Deleter>;

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    std::vector<float> result(n * n);
    size_t byte_size = static_cast<size_t>(n * n * sizeof(float));
    float alpha = 1.0f;
    float beta = 0.0f;

    float *A = nullptr, *B = nullptr, *res = nullptr;
    cudaMalloc(&A, byte_size);
    my_pointer d_A(A);
    cudaMalloc(&B, byte_size);
    my_pointer d_B(B);
    cudaMalloc(&res, byte_size);
    my_pointer d_res(res);
    
    cublasSetVector(n*n,sizeof(float),a.data(),1,A,1);
    cublasSetVector(n*n,sizeof(float),b.data(),1,B,1);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n, &alpha, B, n,A, n, &beta,res,n);
    cublasGetVector(n*n,sizeof(float),res,1,result.data(),1);

    cublasDestroy(handle);
    return result;
}