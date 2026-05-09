#ifdef __GNUC__
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("avx,avx2,fma")
#endif

#include <vector>
#include <cmath>
#include <omp.h>

#include "gelu_omp.h"

/* Optimizations

1. Optimized calculations (sigmoid form)
2. Inlined function into loop
3. Thread parallelism + SIMD for each thread
4. Flags
5. loop-unrolling via flag (should help?)
6. Exponent approximations that should be faster, BUT can harm precision
*/

// Abusing floating-point representation for fast exp approximation
// Precision is too low
/*
Max absolute error: 0.00800818
Best time: 33.8271 ms
*/
inline float fast_exp1(float x) {
    union { float f; int i; } u;
    u.i = (int)(12102203.0f * x) + 127 * (1 << 23);
    return u.f;
}

// Using approximation for exponent (8 multiplications)
// Precision is too low
/*
Max absolute error: 0.00133055
Best time: 33.843 ms
*/
inline float fast_exp2(float x) {
    x = 1.0f + x / 256.0f;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    return x;
}

// 10 multiplications
// Precision is enough??
/*
Max absolute error: 0.00033576
Best time: 37.0399 ms
*/
inline float fast_exp3(float x) {
    x = 1.0f + x / 1024.0f;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x;
    return x;
}

// 12 multiplications
// Precision is enough??
/* Test:
Max absolute error: 0.000106648
Best time: 33.0428 ms
*/
inline float fast_exp4(float x) {
    x = 1.0f + x / 4096.0f;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    return x;
}

constexpr float coeff_1 = 1.59576912f;
constexpr float coeff_2 = 0.044715f;

std::vector<float> GeluOMP(const std::vector<float>& input)
{
    int sz = static_cast<int>(input.size());
    std::vector<float> res(sz);


#ifdef __GNUC__
#pragma omp parallel for simd schedule(static)
#else
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < sz; i++)
    {
        float x = input[i];

        float x3 = x * x * x;
        float exp_arg = coeff_1 * (x + coeff_2 * x3);
        float sigmoid = 1.0f / (1.0f + fast_exp3(-exp_arg));

        res[i] = x * sigmoid;
    }
    return res;
}