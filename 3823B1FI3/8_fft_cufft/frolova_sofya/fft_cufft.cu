#include "fft_cufft.h"
#include <cufft.h>
#include <cuda_runtime.h>

__global__ void normalize(cufftComplex* data, int count, float inv_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx].x *= inv_n;
        data[idx].y *= inv_n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    const size_t totalFloats = input.size();
    const int n = static_cast<int>(totalFloats / (2 * batch));
    const float inv_n = 1.0f / n;   

    cufftComplex* d_data;
    cudaMalloc(&d_data, totalFloats * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

 
    cudaMemcpyAsync(d_data, input.data(), totalFloats * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

  
    cufftHandle plan;
    cufftCreate(&plan);
    cufftSetStream(plan, stream);

    int rank = 1;
    int n_elem = n;
    int inembed[] = {n};
    int onembed[] = {n};
    int istride = 1, ostride = 1;
    int idist = n, odist = n;

    cufftPlanMany(&plan, rank, &n_elem,
                  inembed, istride, idist,
                  onembed, ostride, odist,
                  CUFFT_C2C, batch);

    
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);


    int blocks = (totalFloats / 2 + 255) / 256;
    normalize<<<blocks, 256, 0, stream>>>(d_data, totalFloats / 2, inv_n);

    std::vector<float> result(totalFloats);

    cudaMemcpyAsync(result.data(), d_data, totalFloats * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);


    cufftDestroy(plan);
    cudaStreamDestroy(stream);
    cudaFree(d_data);

    return result;
}