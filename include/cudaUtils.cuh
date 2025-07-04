#ifndef CUDA_UTILS_COVERAGE
#define CUDA_UTILS_COVERAGE

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// CUBLAS error checking macro
#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t status = call;                                             \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "CUBLAS error at %s:%d - status %d\n", __FILE__, __LINE__, \
                    status);                                                      \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#endif // CUDA_UTILS_COVERAGE