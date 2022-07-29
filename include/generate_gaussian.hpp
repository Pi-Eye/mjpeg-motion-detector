#ifndef GENERATE_GAUSSIAN_HPP
#define GENERATE_GAUSSIAN_HPP

#include <vector>

/**
 * Gaussian - Represents a Gaussian Kernal
 *
 * scale:   amount kernal was scaled up by
 * kernel:  kernel itself
 */
struct Gaussian {
  unsigned int scale;
  std::vector<double> kernel;
};

/**
 * GenerateGaussian() - Generates 1D gaussian blur kernel
 *
 * size:      size of gaussian blur (0 mean no blur, 1 means 3x1, 2 means 5x1, etc.)
 * returns:   Gaussian - Gaussian Kernel
 */
Gaussian GenerateGaussian(unsigned int size);

/**
 * ScaleGaussian() - Scales up gaussian blur kernel
 *
 * gaussian:  gaussian kernel to scale up
 * scale:     amount to scale kernel up by
 * returns:   Gaussian - Gaussian Kernel
 */
Gaussian ScaleGaussian(Gaussian& gaussian, unsigned int scale);

#endif