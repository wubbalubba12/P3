#include "gpu_matrix_convolution.h"

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace {

inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
    }
}

__global__ void convolution_kernel(const double* __restrict__ src,
                                   std::uint32_t w,
                                   std::uint32_t h,
                                   const double* __restrict__ kernel,
                                   std::uint32_t kw,
                                   std::uint32_t kh,
                                   double* __restrict__ out) {
    extern __shared__ double sk[]; // shared kernel

    const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int tcount = blockDim.x * blockDim.y;
    const std::size_t kN = static_cast<std::size_t>(kw) * static_cast<std::size_t>(kh);

    for (std::size_t i = tid; i < kN; i += tcount) {
        sk[i] = kernel[i];
    }
    __syncthreads();

    const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int kx_center = static_cast<int>(kw / 2u);
    const int ky_center = static_cast<int>(kh / 2u);

    double sum = 0.0;

    for (std::uint32_t ky = 0; ky < kh; ++ky) {
        for (std::uint32_t kx = 0; kx < kw; ++kx) {
            const int sx = static_cast<int>(x) + (static_cast<int>(kx) - kx_center);
            const int sy = static_cast<int>(y) + (static_cast<int>(ky) - ky_center);

            if (sx < 0 || sy < 0 || sx >= static_cast<int>(w) || sy >= static_cast<int>(h))
                continue; // zero padding

            const std::uint32_t fkx = (kw - 1u) - kx;
            const std::uint32_t fky = (kh - 1u) - ky;

            const std::size_t sidx = static_cast<std::size_t>(sy) * w + static_cast<std::size_t>(sx);
            const std::size_t kidx = static_cast<std::size_t>(fky) * kw + static_cast<std::size_t>(fkx);

            sum += src[sidx] * sk[kidx];
        }
    }

    out[static_cast<std::size_t>(y) * w + x] = sum;
}

}
void matrix_convolution(void** d_source_matrix,
                        std::uint32_t matrix_width,
                        std::uint32_t matrix_height,
                        void** d_kernel,
                        std::uint32_t kernel_width,
                        std::uint32_t kernel_height,
                        void** d_result) {
    if (!d_source_matrix || !*d_source_matrix || !d_kernel || !*d_kernel || !d_result) return;

    const std::size_t n = static_cast<std::size_t>(matrix_width) * static_cast<std::size_t>(matrix_height);
    if (n == 0) { *d_result = nullptr; return; }

    const std::size_t out_bytes = n * sizeof(double);
    cuda_check(cudaMalloc(d_result, out_bytes), "cudaMalloc(d_result)");
    if (*d_result == nullptr) return;

    const dim3 block(16, 16);
    const dim3 grid((matrix_width + block.x - 1) / block.x,
                    (matrix_height + block.y - 1) / block.y);

    const std::size_t shared_bytes =
        static_cast<std::size_t>(kernel_width) * static_cast<std::size_t>(kernel_height) * sizeof(double);

    convolution_kernel<<<grid, block, shared_bytes>>>(
        static_cast<const double*>(*d_source_matrix),
        matrix_width, matrix_height,
        static_cast<const double*>(*d_kernel),
        kernel_width, kernel_height,
        static_cast<double*>(*d_result)
    );

    cuda_check(cudaGetLastError(), "convolution_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}
