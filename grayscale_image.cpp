#include "grayscale_image.h"
#include <iostream>
#include <omp.h>
#include "intermediate_image.h"
#include <algorithm>
#include <cmath>

void GrayscaleImage::convert_bitmap(BitmapImage& bitmap){
    // Adjust dimensions to match the bitmap
    height = bitmap.get_height();
    width = bitmap.get_width();
    pixels.resize(height * width);
    
    // Convert RGB pixels to grayscale using OpenMP parallelization
    #pragma omp parallel for
    for(std::int32_t y = 0; y < static_cast<std::int32_t>(height); ++y){
        for(std::uint32_t x = 0; x < width; ++x){
            auto pixel = bitmap.get_pixel(y, x);
            
            // Get RGB values
            double R = static_cast<double>(pixel.get_red_channel());
            double G = static_cast<double>(pixel.get_green_channel());
            double B = static_cast<double>(pixel.get_blue_channel());
            
            // Calculate luminance: L = 0.2126*R + 0.7152*G + 0.0722*B + 0.5
            double L = 0.2126 * R + 0.7152 * G + 0.0722 * B + 0.5;
            
            // Convert to uint8_t (will truncate the decimal part)
            std::uint8_t grayscale_value = static_cast<std::uint8_t>(L);
            
            // Store in row-major format
            pixels[y * width + x] = grayscale_value;
        }
    }
}

void GrayscaleImage::convert_intermediate_image(IntermediateImage& image){
    // Update min and max pixel values
    image.update_min_pixel_value();
    image.update_max_pixel_value();
    
    // Adjust dimensions to match the intermediate image
    height = image.height;
    width = image.width;
    pixels.resize(height * width);
    
    // Determine normalization range
    double min_val = image.min_pixel_value;
    double max_val = image.max_pixel_value;
    
    // If min and max are within [0, 255], use 0 and 255 for normalization
    if(min_val >= 0.0 && min_val <= 255.0 && max_val >= 0.0 && max_val <= 255.0){
        min_val = 0.0;
        max_val = 255.0;
    }
    
    // Convert double values to uint8_t using normalization
    #pragma omp parallel for
    for(std::int32_t y = 0; y < static_cast<std::int32_t>(height); ++y){
        for(std::uint32_t x = 0; x < width; ++x){
            std::uint32_t idx = y * width + x;
            double v = image.pixels[idx];
            
            // Normalize: g = (v - min) / (max - min) * 255
            std::uint8_t g;
            if(max_val == min_val){
                // Avoid division by zero
                g = 0;
            } else {
                double normalized = ((v - min_val) / (max_val - min_val)) * 255.0;
                
                // Clamp to [0, 255] and convert
                if(normalized < 0.0) normalized = 0.0;
                if(normalized > 255.0) normalized = 255.0;
                
                g = static_cast<std::uint8_t>(normalized);
            }
            
            pixels[idx] = g;
        }
    }
}
