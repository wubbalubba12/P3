#include "gpu_downsampling.h"
#include "gpu_memory_management.h"
#include <cstdio>
#include <cuda_runtime_api.h>
#include <cuda.h>


// CUDA kernel for downsampling
// Each 3x3 block in source maps to one pixel in result
__global__ void downsampling_kernel(const double* d_source, double* d_result,
                                     std::uint32_t source_height, std::uint32_t source_width,
                                     std::uint32_t result_height, std::uint32_t result_width){
    std::uint32_t result_x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t result_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(result_x >= result_width || result_y >= result_height){
        return;
    }
    
    // Calculate the center position in the source image
    std::int32_t center_x = static_cast<std::int32_t>(result_x) * 3;
    std::int32_t center_y = static_cast<std::int32_t>(result_y) * 3;
    
    // Sum all valid pixels in the 3x3 block
    double sum = 0.0;
    std::uint32_t count = 0;
    
    // Iterate over 3x3 block centered at (center_x, center_y)
    for(std::int32_t dy = -1; dy <= 1; ++dy){
        for(std::int32_t dx = -1; dx <= 1; ++dx){
            std::int32_t source_x = center_x + dx;
            std::int32_t source_y = center_y + dy;
            
            // Check if pixel is within source image bounds
            if(source_x >= 0 && source_x < static_cast<std::int32_t>(source_width) &&
               source_y >= 0 && source_y < static_cast<std::int32_t>(source_height)){
                sum += d_source[source_y * source_width + source_x];
                count++;
            }
        }
    }
    
    // Calculate average
    double average = (count > 0) ? (sum / static_cast<double>(count)) : 0.0;
    
    // Write result
    d_result[result_y * result_width + result_x] = average;
}

void image_downsampling(void** d_source_image, std::uint32_t source_image_height,
                        std::uint32_t source_image_width, void** d_result){
    // Calculate result dimensions
    std::uint32_t result_height = (source_image_height + 2) / 3; // Ceiling division
    std::uint32_t result_width = (source_image_width + 2) / 3;
    
    // Allocate memory for result on GPU
    std::size_t result_size = result_height * result_width * sizeof(double);
    cudaMalloc(d_result, result_size);
    
    // Cast void pointers
    double* d_source = static_cast<double*>(*d_source_image);
    double* d_res = static_cast<double*>(*d_result);
    
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((result_width + blockSize.x - 1) / blockSize.x,
                  (result_height + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    downsampling_kernel<<<gridSize, blockSize>>>(d_source, d_res,
                                                  source_image_height, source_image_width,
                                                  result_height, result_width);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
}

std::uint32_t get_downsampled_width(std::uint32_t image_width){
    return (image_width + 2) / 3;
}

std::uint32_t get_downsampled_height(std::uint32_t image_height){
    return (image_height + 2) / 3;
}
