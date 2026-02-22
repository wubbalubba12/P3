#include "gpu_memory_management.h"
#include <cuda_runtime_api.h>
#include <cuda.h>


void allocate_device_memory(IntermediateImage& image, void** devPtr){
    // Calculate size needed for image pixels
    std::size_t size = image.height * image.width * sizeof(double);
    
    // Allocate memory on GPU
    cudaMalloc(devPtr, size);
}

void free_device_memory(void** devPtr){
    if(devPtr != nullptr && *devPtr != nullptr){
        // Free the GPU memory
        cudaFree(*devPtr);
        
        // Set pointer to nullptr
        *devPtr = nullptr;
    }
}

void copy_data_to_device(IntermediateImage& image, void** devPtr){
    // If devPtr is nullptr, allocate memory first
    if(*devPtr == nullptr){
        allocate_device_memory(image, devPtr);
    }
    
    // Calculate size
    std::size_t size = image.height * image.width * sizeof(double);
    
    // Copy data from host to device
    cudaMemcpy(*devPtr, image.pixels.data(), size, cudaMemcpyHostToDevice);
}

void copy_data_from_device(void** devPtr, IntermediateImage& image){
    if(devPtr != nullptr && *devPtr != nullptr){
        // Calculate size
        std::size_t size = image.height * image.width * sizeof(double);
        
        // Ensure image.pixels is properly sized
        image.pixels.resize(image.height * image.width);
        
        // Copy data from device to host
        cudaMemcpy(image.pixels.data(), *devPtr, size, cudaMemcpyDeviceToHost);
    }
}
