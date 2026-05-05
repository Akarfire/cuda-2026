#include "block_gemm_omp.h"
#include <algorithm>
#include <cstring>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

    const int block_size = (n < 64) ? n : 64;
    const int max_bs = 64;                
    const int local_size = max_bs * max_bs; 

    // Динамическое планирование – потоки берут блоки по мере готовности
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            // Локальный аккумулятор (на стеке) – избегаем false sharing
            float local_c[local_size] = {};

            for (int k = 0; k < n; k += block_size) {
                for (int ii = 0; ii < block_size; ++ii) {
                    for (int kk = 0; kk < block_size; ++kk) {
                        float aik = a[(i + ii) * n + (k + kk)];
                        // Векторизованное умножение-сложение строки блока B
                        #pragma omp simd
                        for (int jj = 0; jj < block_size; ++jj) {
                            local_c[ii * block_size + jj] +=
                                aik * b[(k + kk) * n + (j + jj)];
                        }
                    }
                }
            }

            // Запись посчитанного блока в глобальную матрицу C
            for (int ii = 0; ii < block_size; ++ii) {
                for (int jj = 0; jj < block_size; ++jj) {
                    c[(i + ii) * n + (j + jj)] = local_c[ii * block_size + jj];
                }
            }
        }
    }

    return c;
}
