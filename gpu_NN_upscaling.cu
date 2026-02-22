#include "gpu_NN_upscaling.h"
#include "gpu_memory_management.h"
#include <cstdio>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>

// CUDA kernel for nearest-neighbor upscaling
// Each pixel in source becomes a 3x3 block in result
__global__ void nn_upscaling_kernel(const double* d_source, double* d_result, 
                                     std::uint32_t source_height, std::uint32_t source_width,
                                     std::uint32_t result_height, std::uint32_t result_width){
    // Calculate which pixel in the result image this thread handles
    std::uint32_t result_x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t result_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(result_x >= result_width || result_y >= result_height){
        return;
    }
    
    // Map back to source image coordinates
    // Each source pixel maps to a 3x3 block
    std::uint32_t source_x = result_x / 3;
    std::uint32_t source_y = result_y / 3;
    
    // Get the source pixel value
    double pixel_value = d_source[source_y * source_width + source_x];
    
    // Write to result
    d_result[result_y * result_width + result_x] = pixel_value;
}

void NN_image_upscaling(void** d_source_image, std::uint32_t source_image_height, 
                        std::uint32_t source_image_width, void** d_result){
    // Calculate result dimensions (3x upscaling)
    std::uint32_t result_height = source_image_height * 3;
    std::uint32_t result_width = source_image_width * 3;
    
    // Allocate memory for result on GPU
    std::size_t result_size = result_height * result_width * sizeof(double);
    cudaMalloc(d_result, result_size);
    
    // Cast void pointers to double pointers
    double* d_source = static_cast<double*>(*d_source_image);
    double* d_res = static_cast<double*>(*d_result);
    
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((result_width + blockSize.x - 1) / blockSize.x,
                  (result_height + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    nn_upscaling_kernel<<<gridSize, blockSize>>>(d_source, d_res,
                                                  source_image_height, source_image_width,
                                                  result_height, result_width);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
}

std::uint32_t get_NN_upscaled_width(std::uint32_t image_width){
    return image_width * 3;
}

std::uint32_t get_NN_upscaled_height(std::uint32_t image_height){
    return image_height * 3;
}
