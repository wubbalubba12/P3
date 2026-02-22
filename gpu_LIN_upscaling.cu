#include "gpu_LIN_upscaling.h"
#include "gpu_memory_management.h"
#include <cstdio>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
// Kernel to distribute original pixels in the upscaled image
__global__ void distribute_pixels_kernel(const double* d_source, double* d_result,
                                         std::uint32_t source_height, std::uint32_t source_width,
                                         std::uint32_t result_height, std::uint32_t result_width){
    std::uint32_t source_x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t source_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(source_x >= source_width || source_y >= source_height){
        return;
    }
    
    // Each source pixel goes to position (2*y, 2*x) in result
    std::uint32_t result_x = source_x * 2;
    std::uint32_t result_y = source_y * 2;
    
    double pixel_value = d_source[source_y * source_width + source_x];
    d_result[result_y * result_width + result_x] = pixel_value;
}

// Kernel to interpolate along X and Y axes
__global__ void interpolate_axes_kernel(double* d_result,
                                        std::uint32_t result_height, std::uint32_t result_width){
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x >= result_width || y >= result_height){
        return;
    }
    
    // Interpolate horizontal gaps (odd x, even y)
    if(x % 2 == 1 && y % 2 == 0){
        if(x > 0 && x < result_width - 1){
            double left = d_result[y * result_width + (x - 1)];
            double right = d_result[y * result_width + (x + 1)];
            d_result[y * result_width + x] = (left + right) / 2.0;
        }
    }
    
    // Interpolate vertical gaps (even x, odd y)
    if(x % 2 == 0 && y % 2 == 1){
        if(y > 0 && y < result_height - 1){
            double top = d_result[(y - 1) * result_width + x];
            double bottom = d_result[(y + 1) * result_width + x];
            d_result[y * result_width + x] = (top + bottom) / 2.0;
        }
    }
}

// Kernel to interpolate diagonal positions
__global__ void interpolate_diagonals_kernel(double* d_result,
                                             std::uint32_t result_height, std::uint32_t result_width){
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x >= result_width || y >= result_height){
        return;
    }
    
    // Interpolate diagonal positions (odd x, odd y)
    if(x % 2 == 1 && y % 2 == 1){
        if(x > 0 && x < result_width - 1 && y > 0 && y < result_height - 1){
            double top_left = d_result[(y - 1) * result_width + (x - 1)];
            double top_right = d_result[(y - 1) * result_width + (x + 1)];
            double bottom_left = d_result[(y + 1) * result_width + (x - 1)];
            double bottom_right = d_result[(y + 1) * result_width + (x + 1)];
            
            d_result[y * result_width + x] = (top_left + top_right + bottom_left + bottom_right) / 4.0;
        }
    }
}

void LIN_image_upscaling(void** d_source_image, std::uint32_t source_image_height,
                         std::uint32_t source_image_width, void** d_result){
    // Calculate result dimensions: 2*n - 1 for each dimension
    std::uint32_t result_height = source_image_height * 2 - 1;
    std::uint32_t result_width = source_image_width * 2 - 1;
    
    // Allocate memory for result on GPU
    std::size_t result_size = result_height * result_width * sizeof(double);
    cudaMalloc(d_result, result_size);
    
    // Initialize result to zero
    cudaMemset(*d_result, 0, result_size);
    
    // Cast void pointers
    double* d_source = static_cast<double*>(*d_source_image);
    double* d_res = static_cast<double*>(*d_result);
    
    // Step 1: Distribute original pixels
    dim3 blockSize1(16, 16);
    dim3 gridSize1((source_image_width + blockSize1.x - 1) / blockSize1.x,
                   (source_image_height + blockSize1.y - 1) / blockSize1.y);
    
    distribute_pixels_kernel<<<gridSize1, blockSize1>>>(d_source, d_res,
                                                         source_image_height, source_image_width,
                                                         result_height, result_width);
    cudaDeviceSynchronize();
    
    // Step 2: Interpolate along X and Y axes
    dim3 blockSize2(16, 16);
    dim3 gridSize2((result_width + blockSize2.x - 1) / blockSize2.x,
                   (result_height + blockSize2.y - 1) / blockSize2.y);
    
    interpolate_axes_kernel<<<gridSize2, blockSize2>>>(d_res, result_height, result_width);
    cudaDeviceSynchronize();
    
    // Step 3: Interpolate diagonals
    interpolate_diagonals_kernel<<<gridSize2, blockSize2>>>(d_res, result_height, result_width);
    cudaDeviceSynchronize();
}

std::uint32_t get_LIN_upscaled_width(std::uint32_t image_width){
    return image_width * 2 - 1;
}

std::uint32_t get_LIN_upscaled_height(std::uint32_t image_height){
    return image_height * 2 - 1;
}
