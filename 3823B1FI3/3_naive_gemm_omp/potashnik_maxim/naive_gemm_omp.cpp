#ifdef __GNUC__
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("avx,avx2,fma")
#endif

/* Optimizations
1. Parralelism, SIMD, Flags
2. Better cache usage (IJK -> IKJ)
3. Calculating a[i][k] out of j-loop
4. SIMD in inner loop
5. Using pointers to (MAYBE) simplify data access (and help SIMD)
*/

#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> c(n * n);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float a_ik = a[i * n + k]; 
            float* c_i = &c[i * n];
            const float* b_k = &b[k * n];

#ifdef __GNUC__
#pragma omp simd
#endif
            for (int j = 0; j < n; j++) {
                c_i[j] += a_ik * b_k[j];
            }
        }
    }

    return c;
}