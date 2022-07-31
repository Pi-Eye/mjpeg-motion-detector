#include <cmath>
#include <iostream>
#include <limits>

#include "generate_gaussian.hpp"

bool CompareKernels(std::vector<double> kernel_A, std::vector<double> kernel_B) {
  // Check that sizes match
  bool match = kernel_A.size() == kernel_B.size();
  if (kernel_A.size() != kernel_B.size()) {
    std::cout << "Kernels do not have same size" << std::endl;
    return false;
  }

  // Check that values match
  double sum = 0;
  for (int i = 0; i < kernel_A.size(); i++) {
    if (abs(kernel_A.at(i) - kernel_B.at(i)) > 0.01) {
      std::cout << "Kernel does not match at index: " << i << std::endl;
      match = false;
    }
    sum += kernel_B.at(i);
  }

  // Check that sum is equal to 1
  if (abs(sum - 1) > 0.001) {
    std::cout << "Kernel sum is not 1" << std::endl;
    match = false;
  }

  return match;
}

TEST_CASE("Generate kernels With 1x Scale") {
  SECTION("1x1 kernel") { 
    std::vector<double> solution_kernel = {1};

    Gaussian gaussian = GenerateGaussian(0);

    REQUIRE(CompareKernels(solution_kernel, gaussian.kernel));
  }
  SECTION("3x3 kernel") {
    std::vector<double> solution_kernel = {0.274, 0.452, 0.274};

    Gaussian gaussian = GenerateGaussian(1);

    REQUIRE(CompareKernels(solution_kernel, gaussian.kernel));
  }
  SECTION("5x5 kernel") {
    std::vector<double> solution_kernel = {0.054, 0.244, 0.403, 0.244, 0.054};

    Gaussian gaussian = GenerateGaussian(2);

    REQUIRE(CompareKernels(solution_kernel, gaussian.kernel));
  }
}

TEST_CASE("Generate kernels With 2x Scale") {
  SECTION("1x1 kernel") {
    std::vector<double> solution_kernel = {1 / 2.0, 1 / 2.0};

    Gaussian gaussian = GenerateGaussian(0);
    gaussian = ScaleGaussian(gaussian, 2);

    REQUIRE(CompareKernels(solution_kernel, gaussian.kernel));
  }
  SECTION("3x3 kernel") {
    std::vector<double> solution_kernel = {0.274 / 2.0, 0.274 / 2.0, 0.452 / 2.0, 0.452 / 2.0, 0.274 / 2.0, 0.274 / 2.0};

    Gaussian gaussian = GenerateGaussian(1);
    gaussian = ScaleGaussian(gaussian, 2);

    REQUIRE(CompareKernels(solution_kernel, gaussian.kernel));
  }
  SECTION("5x5 kernel") {
    std::vector<double> solution_kernel = {0.054 / 2.0, 0.054 / 2.0, 0.244 / 2.0, 0.244 / 2.0, 0.403 / 2.0, 0.403 / 2.0, 0.244 / 2.0, 0.244 / 2.0, 0.054 / 2.0, 0.054 / 2.0};

    Gaussian gaussian = GenerateGaussian(2);
    gaussian = ScaleGaussian(gaussian, 2);

    REQUIRE(CompareKernels(solution_kernel, gaussian.kernel));
  }
}