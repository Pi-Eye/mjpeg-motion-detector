#include "motion_detector.hpp"

#include <CL/cl.hpp>
#include <stdexcept>

#include "generate_gaussian.hpp"
#include "open_cl_interface.hpp"

MotionDetector::MotionDetector(InputVideoSettings input_vid_settings, MotionConfig motion_config, DeviceConfig device_config)
    : input_vid_(input_vid_settings), motion_config_(motion_config), device_config_(device_config), opencl_(OpenCLInterface(device_config)) {
  // Check if scale denominator is 0 and throw error if it is
  if (motion_config.scale_denominator == 0) throw std::invalid_argument("Scale denominator cannot be 0");

  // Check if stabilize background and movement is 0 and throw error if it is
  if (motion_config.bg_stabil_length == 0) throw std::invalid_argument("Background stabilization length cannot be 0");
  if (motion_config.motion_stabil_length == 0) throw std::invalid_argument("Movement stabilization length cannot be 0");

  // Check if miniumn changed pixels is not negative and also not greater than 1
  if (motion_config.min_changed_pixels < 0) throw std::invalid_argument("Minimum changed pixels cannot be negative");
  if (motion_config.min_changed_pixels > 1) throw std::invalid_argument("Minimum changed pixels cannot be gretaer than 1");

  // Create gaussian blur
  std::vector<double> gaussian = GenerateGaussian(motion_config_.gaussian_size);
  gaussian = ScaleGaussian(gaussian, motion_config_.scale_denominator);

  // Convert gaussian to OpenCL buffer
  // Convert to double array
  double* temp_gaussian = new double[gaussian.size()];
  for (int i = 0; i < gaussian.size(); i++) temp_gaussian[i] = gaussian.at(i);
  // Move to GPU
  int error = 0;
  gaussian_kernel_ = cl::Buffer(opencl_.GetContext(), CL_MEM_READ_ONLY, gaussian.size() * sizeof(float), nullptr, &error);
  if (error != 0) throw std::runtime_error("Error writing gaussian kernel buffer to OpenCL device");
  // dealloc double array
  delete[] temp_gaussian;

  // Check height and width and throw error if too small
  if (input_vid_.width < gaussian.size()) throw std::invalid_argument("Input video width is too small!");
  if (input_vid_.height < gaussian.size()) throw std::invalid_argument("Input video height is too small!");
}

MotionDetector::~MotionDetector() {}

bool MotionDetector::DetectOnFrame(unsigned char* frame) {
  opencl_.GetContext();
  return false;
}

cl::Buffer BlurAndScale(unsigned char* image);

StabilizedFrames StabilizeFrames();

InputVideoSettings MotionDetector::GetInputVideoSettings() const { return input_vid_; }

MotionConfig MotionDetector::GetMotionConfig() const { return motion_config_; }

DeviceConfig MotionDetector::GetDeviceConfig() const { return device_config_; }

cl::CommandQueue& MotionDetector::GetCLQueue() { return opencl_.GetCommandQueue(); }
