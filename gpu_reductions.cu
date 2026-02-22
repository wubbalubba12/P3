#include "gpu_reductions.h"
#include <cstdio>
#include <iostream>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <cuda_runtime_api.h>

namespace {

inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
    }
}

__device__ __forceinline__ double dmax(double a, double b) { return a > b ? a : b; }

template <unsigned int BLOCK_SIZE>
__global__ void max_reduce_kernel(const double* __restrict__ in, std::size_t n,
                                  double* __restrict__ out) {
    __shared__ double sdata[BLOCK_SIZE];

    const unsigned int tid = threadIdx.x;
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * (BLOCK_SIZE * 2ULL) + tid;

    double v = -DBL_MAX;
    if (start < n) {
        v = in[start];
        const std::size_t start2 = start + BLOCK_SIZE;
        if (start2 < n) v = dmax(v, in[start2]);
    }

    sdata[tid] = v;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = dmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

} 

double get_max_value(void** d_source_image,
                     std::uint32_t source_image_height,
                     std::uint32_t source_image_width) {
    if (d_source_image == nullptr || *d_source_image == nullptr) return 0.0;

    const std::size_t n =
        static_cast<std::size_t>(source_image_height) * static_cast<std::size_t>(source_image_width);
    if (n == 0) return 0.0;

    constexpr unsigned int BLOCK = 256;
    const double* d_in = static_cast<const double*>(*d_source_image);

    std::size_t cur_n = n;
    const double* cur_in = d_in;
    bool cur_in_is_source = true;

    while (cur_n > 1) {
        const std::size_t blocks = (cur_n + (BLOCK * 2ULL - 1ULL)) / (BLOCK * 2ULL);
        const std::size_t bytes = blocks * sizeof(double);

        double* d_out = nullptr;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_out), bytes), "cudaMalloc(reduction buffer)");
        if (d_out == nullptr) {
            if (!cur_in_is_source) cudaFree(const_cast<double*>(cur_in));
            return 0.0;
        }

        max_reduce_kernel<BLOCK><<<static_cast<unsigned int>(blocks), BLOCK>>>(cur_in, cur_n, d_out);
        cuda_check(cudaGetLastError(), "max_reduce_kernel launch");
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        if (!cur_in_is_source) cuda_check(cudaFree(const_cast<double*>(cur_in)), "cudaFree(intermediate)");

        cur_in_is_source = false;
        cur_in = d_out;
        cur_n = blocks;
    }

    double result = 0.0;
    cuda_check(cudaMemcpy(&result, cur_in, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(result)");

    if (!cur_in_is_source) cuda_check(cudaFree(const_cast<double*>(cur_in)), "cudaFree(final)");

    return result;
}
