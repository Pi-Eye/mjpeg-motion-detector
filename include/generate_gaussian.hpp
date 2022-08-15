#ifndef GENERATE_GAUSSIAN_HPP
#define GENERATE_GAUSSIAN_HPP

#include <vector>

/**
 * GenerateGaussian() - Generates 1D gaussian blur kernel
 *
 * size:      size of gaussian blur (0 mean no blur, 1 means 3x1, 2 means 5x1, etc.)
 * returns:   Gaussian - Gaussian Kernel
 */
std::vector<double> GenerateGaussian(unsigned int size);

/**
 * ScaleGaussian() - Scales up gaussian blur kernel
 *
 * gaussian:  gaussian kernel to scale up
 * scale:     amount to scale kernel up by
 * returns:   Gaussian - Gaussian Kernel
 */
std::vector<double> ScaleGaussian(std::vector<double>& gaussian, unsigned int scale);

#endif