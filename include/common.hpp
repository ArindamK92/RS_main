#pragma once
#include <iostream>
#include <cuda_runtime.h>

constexpr long long THREADS_PER_BLOCK = 1024;

#define CUDA_CHECK(call)                                                    \
{                                                                           \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess)                                                 \
    {                                                                       \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__;       \
        std::cerr << " code=" << err << " (" << cudaGetErrorString(err) << ")" << std::endl; \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
}

#define CUDA_ERROR_CHECK
#define CUDASAFECALL(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CUDACHECKERROR()  __cudaCheckError(__FILE__, __LINE__)

inline void __cudaSafeCall(cudaError_t err, const char* file, const long long line)
{
#ifdef CUDA_ERROR_CHECK
    if (err != cudaSuccess)
    {
        std::cerr << "cudaSafeCall() failed at " << file << ":" << line
                  << " : " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
#endif
}

inline void __cudaCheckError(const char* file, const long long line)
{
#ifdef CUDA_ERROR_CHECK
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "cudaCheckError() failed at " << file << ":" << line
                  << " : " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
#endif
}

// --------------------------------------------------
// Debug Print
// --------------------------------------------------
#ifdef DEBUG
    #define DEBUG_PRINT(x) std::cout << x << std::endl
#else
    #define DEBUG_PRINT(x)
#endif
