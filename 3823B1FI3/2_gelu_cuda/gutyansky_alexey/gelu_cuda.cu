#include "gelu_cuda.h"

#include <vector>
#include <cmath>

__global__ void GeluCUDA_cu(float *mem, int N) {
    const float _2SQRT2PI = 2.0f * sqrtf(2.0f / 3.141592653589793238462643f);
    const float C1 = 0.044715f;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = mem[i];

        float x3 = x * x * x;
        float arg = _2SQRT2PI * (x + C1 * x3);
        float ex = expf(-arg);

        mem[i] = x / (1.0f + ex);
    }

}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int N = input.size();
    const int ByteSize = sizeof(float) * N;
    const int BlockSize = 256;
    
    static float *mem_cu = nullptr;
    static int mem_sz = 0;
    if (mem_sz != ByteSize) {
        cudaFree(mem_cu);
        cudaMalloc(&mem_cu, ByteSize);
        mem_sz = ByteSize;
    }
    
    cudaMemcpy(mem_cu, input.data(), ByteSize, cudaMemcpyHostToDevice);
    
    int num_blocks = (N + BlockSize - 1) / BlockSize;
    GeluCUDA_cu<<<num_blocks,BlockSize>>>(mem_cu, N);
    
    std::vector<float> output(N);
    cudaMemcpy(output.data(), mem_cu, ByteSize, cudaMemcpyDeviceToHost);

    return output;
}