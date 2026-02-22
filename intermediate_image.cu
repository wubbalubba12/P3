#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

#include "intermediate_image.h"
#include "gpu_matrix_convolution.h"

namespace {

inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
    }
}

__global__ void magnitude_kernel(const double* __restrict__ gx,
                                 const double* __restrict__ gy,
                                 double* __restrict__ out,
                                 std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double a = gx[i];
    const double b = gy[i];
    out[i] = sqrt(a * a + b * b);
}

} 

void IntermediateImage::apply_sobel_filter() {
    if (height == 0 || width == 0) return;

    const std::size_t n = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    const std::size_t bytes = n * sizeof(double);

    const double h_kx[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    const double h_ky[9] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };

    void* d_img = nullptr;
    void* d_kx  = nullptr;
    void* d_ky  = nullptr;
    void* d_gx  = nullptr;
    void* d_gy  = nullptr;
    void* d_mag = nullptr;

    cuda_check(cudaMalloc(&d_img, bytes), "cudaMalloc(d_img)");
    cuda_check(cudaMemcpy(d_img, pixels.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(img H2D)");

    cuda_check(cudaMalloc(&d_kx, 9 * sizeof(double)), "cudaMalloc(d_kx)");
    cuda_check(cudaMalloc(&d_ky, 9 * sizeof(double)), "cudaMalloc(d_ky)");
    cuda_check(cudaMemcpy(d_kx, h_kx, 9 * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(kx H2D)");
    cuda_check(cudaMemcpy(d_ky, h_ky, 9 * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(ky H2D)");

    matrix_convolution(&d_img, width, height, &d_kx, 3, 3, &d_gx);
    matrix_convolution(&d_img, width, height, &d_ky, 3, 3, &d_gy);

    cuda_check(cudaMalloc(&d_mag, bytes), "cudaMalloc(d_mag)");
    {
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned int>((n + block.x - 1) / block.x));
        magnitude_kernel<<<grid, block>>>(
            static_cast<const double*>(d_gx),
            static_cast<const double*>(d_gy),
            static_cast<double*>(d_mag),
            n
        );
        cuda_check(cudaGetLastError(), "magnitude_kernel launch");
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    cuda_check(cudaMemcpy(pixels.data(), d_mag, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(result D2H)");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize after D2H");

    if (d_mag) cuda_check(cudaFree(d_mag), "cudaFree(d_mag)");
    if (d_gx)  cuda_check(cudaFree(d_gx),  "cudaFree(d_gx)");
    if (d_gy)  cuda_check(cudaFree(d_gy),  "cudaFree(d_gy)");
    if (d_kx)  cuda_check(cudaFree(d_kx),  "cudaFree(d_kx)");
    if (d_ky)  cuda_check(cudaFree(d_ky),  "cudaFree(d_ky)");
    if (d_img) cuda_check(cudaFree(d_img), "cudaFree(d_img)");
}
