#include "generate_gaussian.hpp"

#include <cmath>
#include <vector>

std::vector<double> GenerateGaussian(unsigned int size) {
  // Calculate Kernal height and width
  unsigned int kernel_size = (2 * size) + 1;

  // Create empty 1D array for kernal
  std::vector<double> kernel;

  // Calculate Kernel Values
  double sigma = 1;
  double sum = 0;
  int center = static_cast<int>(kernel_size) / 2;
  for (int i = 0; i < kernel_size; i++) {
    kernel.push_back(exp(-1 * ((i - center) * (i - center)) / (2 * sigma * sigma)));
    sum += kernel.at(i);
  }

  // Normalize Values
  for (int i = 0; i < kernel_size; i++) kernel.at(i) /= sum;

  return kernel;
}

std::vector<double> ScaleGaussian(std::vector<double>& gaussian, unsigned int scale) {
  // Create resized 1D array for kernel
  std::vector<double> kernel;

  // Copy values over and scale
  double sum = 0;
  for (int i = 0; i < gaussian.size(); i++) {
    for (int j = 0; j < scale; j++) {
      unsigned int index = i * scale + j;
      kernel.push_back(gaussian.at(i) / scale);
    }
  }

  return kernel;
}