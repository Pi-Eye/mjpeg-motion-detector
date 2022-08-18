#include "motion_detector.hpp"

#include <CL/cl.h>

#include <fstream>
#include <iostream>
#include <stdexcept>

#include "generate_gaussian.hpp"
#include "open_cl_interface.hpp"

#define OPEN_CL_COMPILE_FLAGS "-cl-fast-relaxed-math -w"
#define MAX_WORK_GROUP_SIZE 1024

const std::string kBlurScaleVerticalFile = "blur_and_scale_vertical.cl";
const std::string kBlurScaleHorizontalFile = "blur_and_scale_horizontal.cl";
const std::string kStabilizeFile = "stabilize_bg_mvt.cl";
const std::string kCalculateDifferenceFile = "calculate_difference.cl";

MotionDetector::MotionDetector(InputVideoSettings input_vid_settings, MotionConfig motion_config, DeviceConfig device_config)
    : input_vid_(input_vid_settings), motion_config_(motion_config), device_config_(device_config) {
  // Check settings
  ValidateSettings();
  // Calculate Buffer Sizes
  CalculateBufferSizes();

  // Setup OpenCL device
  InitOpenCL();

  // Load Buffers
  LoadBlurAndScaleBuffers();
  LoadStabilizeAndCompareBuffers();
  //  Load kernels
  LoadBlurAndScaleKernels();
  LoadStabilizeAndCompareKernel();

  // Create work sizes
  InitWorkSizes();
}

MotionDetector::~MotionDetector() {
  for (int i = 0; i < frames_.size(); i++) {
    delete[] frames_.at(i);
  }
  frames_.clear();
}

bool MotionDetector::DetectOnFrame(unsigned char* frame) {
  // Run processing kernels
  BlurAndScale(frame);
  StabilizeAndCompareFrames();

  // Pull difference frame from memory
  bool* difference = new bool[scaled_frame_buffer_size_];
  int error = cmd_queue_.enqueueReadBuffer(difference_frame_, CL_TRUE, 0, scaled_frame_buffer_size_ * sizeof(bool), static_cast<void*>(difference));
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to read difference frame from memory with error code: " + std::to_string(error));

  // Sum the total difference
  unsigned int total_diff = 0;
  for (int i = 0; i < scaled_frame_buffer_size_; i++) {
    if (difference[i]) total_diff++;
  }

  return total_diff > diff_threshold_;
}

cl::Buffer& MotionDetector::BlurAndScale(unsigned char* frame) {
  int error = CL_SUCCESS;
  // Write new frame to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(input_frame_, CL_TRUE, 0, input_frame_buffer_size_ * sizeof(unsigned char), static_cast<void*>(frame));

  // Queue kernels
  // Vertical Scale
  error = cmd_queue_.enqueueNDRangeKernel(bs_vertical_kernel_, cl::NullRange, intermediate_scaled_global_work_size_2d_, motion_thread_block_size_2d_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to queue OpenCL kernel with error code: " + std::to_string(error));
  error = cmd_queue_.finish();
  if (error != CL_SUCCESS) throw std::runtime_error("Error while running vertical blur and scale kernel with error code: " + std::to_string(error));

  // Horizontal scale
  error = cmd_queue_.enqueueNDRangeKernel(bs_horizontal_kernel_, cl::NullRange, scaled_global_work_size_2d_, motion_thread_block_size_2d_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to queue OpenCL kernel with error code: " + std::to_string(error));
  error = cmd_queue_.finish();
  if (error != CL_SUCCESS) throw std::runtime_error("Error while running vertical blur and scale kernel with error code: " + std::to_string(error));

  // Read newly scaled frame back to cpu
  unsigned char* scaled_frame = new unsigned char[scaled_frame_buffer_size_];
  error = cmd_queue_.enqueueReadBuffer(scaled_frame_, CL_TRUE, 0, scaled_frame_buffer_size_ * sizeof(unsigned char), static_cast<void*>(scaled_frame));
  if (error != CL_SUCCESS) throw std::runtime_error("Error while reading scaled frame with error code: " + std::to_string(error));

  // Find location for newest frame and delete frame already at that location
  newest_frame_loc_ = (newest_frame_loc_ + 1) % frames_.size();
  delete[] frames_[newest_frame_loc_];
  // Place newly scaled frame in that location
  frames_[newest_frame_loc_] = scaled_frame;

  return scaled_frame_;
}

cl::Buffer& MotionDetector::StabilizeAndCompareFrames() {
  int error = CL_SUCCESS;
  // Write background frame to remove to OpenCL device
  // Determine location of the background frame to remove in list of frames
  bg_remove_loc_ = (bg_remove_loc_ + 1) % frames_.size();
  // Write to device
  error = cmd_queue_.enqueueWriteBuffer(bg_frame_to_remove_, CL_TRUE, 0, scaled_frame_buffer_size_ * sizeof(unsigned char), static_cast<void*>(frames_.at(bg_remove_loc_)));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing background to remove buffer with error code: " + std::to_string(error));

  // Write movement frame to remove to OpenCL device
  mvt_remove_loc_ = (mvt_remove_loc_ + 1) % frames_.size();
  // Write to device
  error = cmd_queue_.enqueueWriteBuffer(mvt_frame_to_remove_, CL_TRUE, 0, scaled_frame_buffer_size_ * sizeof(unsigned char), static_cast<void*>(frames_.at(mvt_remove_loc_)));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing background to remove buffer with error code: " + std::to_string(error));

  // Queue kernel
  error = cmd_queue_.enqueueNDRangeKernel(stabilize_kernel_, cl::NullRange, scaled_global_work_size_1d_, motion_thread_block_size_1d_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to queue OpenCL kernel with error code: " + std::to_string(error));
  error = cmd_queue_.finish();
  if (error != CL_SUCCESS) throw std::runtime_error("Error while running vertical blur and scale kernel with error code: " + std::to_string(error));

  return difference_frame_;
}

void MotionDetector::ValidateSettings() const {
  // Check if scale denominator is 0 and throw error if it is
  if (motion_config_.scale_denominator == 0) throw std::invalid_argument("Scale denominator cannot be 0");

  // Check if stabilize background and movement is 0 and throw error if it is
  if (motion_config_.bg_stabil_length == 0) throw std::invalid_argument("Background stabilization length cannot be 0");
  if (motion_config_.motion_stabil_length == 0) throw std::invalid_argument("Movement stabilization length cannot be 0");

  // Check if miniumn changed pixels is not negative and also not greater than 1
  if (motion_config_.min_changed_pixels < 0) throw std::invalid_argument("Minimum changed pixels cannot be negative");
  if (motion_config_.min_changed_pixels > 1) throw std::invalid_argument("Minimum changed pixels cannot be gretaer than 1");

  // Check height and width of input video and throw error if too small
  std::vector<double> gaussian = GenerateGaussian(motion_config_.gaussian_size);
  gaussian = ScaleGaussian(gaussian, motion_config_.scale_denominator);
  if (input_vid_.width < gaussian.size()) throw std::invalid_argument("Input video width is too small!");
  if (input_vid_.height < gaussian.size()) throw std::invalid_argument("Input video height is too small!");
}

void MotionDetector::InitOpenCL() {
  // Select device
  device_ = OpenCLInterface::GetDevice(device_config_);
  std::cout << "Selected device: " + device_.getInfo<CL_DEVICE_NAME>() << std::endl;
  // Create context and command queue
  int error = CL_SUCCESS;
  context_ = cl::Context(device_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL context with error code: " + std::to_string(error));
  cmd_queue_ = cl::CommandQueue(context_, device_);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating OpenCL command queue with error code: " + std::to_string(error));
}

void MotionDetector::InitWorkSizes() {
  // Calculate maximun thread block size possible
  unsigned int max_thread_block_size_x = 1;
  for (unsigned int i = 1; i < device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() && i < MAX_WORK_GROUP_SIZE; i++) {
    if (scaled_width_ % i == 0 && input_vid_.width) max_thread_block_size_x = i;
  }
  unsigned int max_thread_block_size_y = 1;
  for (unsigned int i = 1; i < device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() && i < MAX_WORK_GROUP_SIZE; i++) {
    if (scaled_height_ % i == 0) max_thread_block_size_y = i;
  }
  std::cout << "Running motion detection with 2D thread block size: " << max_thread_block_size_x << "x" << max_thread_block_size_y << std::endl;
  unsigned int max_thread_block_size = 1;
  for (unsigned int i = 1; i < device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() && i < MAX_WORK_GROUP_SIZE; i++) {
    if ((scaled_height_ * scaled_width_) % i == 0) max_thread_block_size = i;
  }
  std::cout << "Running motion detection with 1D thread block size: " << max_thread_block_size << std::endl;
  // Create 2D ranges
  motion_thread_block_size_2d_ = cl::NDRange(max_thread_block_size_x, max_thread_block_size_y);
  intermediate_scaled_global_work_size_2d_ = cl::NDRange(input_vid_.width, scaled_height_);
  scaled_global_work_size_2d_ = cl::NDRange(scaled_width_, scaled_height_);
  // Create 1D ranges
  motion_thread_block_size_1d_ = cl::NDRange(max_thread_block_size);
  scaled_global_work_size_1d_ = cl::NDRange(static_cast<unsigned int>(scaled_width_ * scaled_height_));
}

void MotionDetector::CalculateBufferSizes() {
  // Remove margin from image for gaussian blur
  unsigned int width_margin_removed = input_vid_.width - (2 * motion_config_.gaussian_size * motion_config_.scale_denominator);
  unsigned int height_margin_removed = input_vid_.height - (2 * motion_config_.gaussian_size * motion_config_.scale_denominator);
  // Calculate scaled width and height
  scaled_width_ = width_margin_removed / motion_config_.scale_denominator;
  scaled_height_ = height_margin_removed / motion_config_.scale_denominator;
  std::cout << "Scaled frame resolution: " << scaled_width_ << "x" << scaled_height_ << std::endl;

  // Calculate buffer sizes
  input_frame_buffer_size_ = input_vid_.width * input_vid_.height;
  if (input_vid_.frame_format == DecompFrameFormat::kRGB) input_frame_buffer_size_ *= 3;  // If RGB frames, need 3 times the bytes
  intermediate_scaled_frame_buffer_size_ = input_vid_.width * scaled_height_;
  scaled_frame_buffer_size_ = scaled_width_ * scaled_height_;

  // Calcualte number of pixels that need to change
  diff_threshold_ = static_cast<unsigned int>(motion_config_.min_changed_pixels * static_cast<double>(scaled_width_ * scaled_height_));
}

void MotionDetector::LoadBlurAndScaleBuffers() {
  // Create buffers
  int error = CL_SUCCESS;
  // gaussian
  std::vector<double> gaussian = GenerateGaussian(motion_config_.gaussian_size);
  gaussian = ScaleGaussian(gaussian, motion_config_.scale_denominator);
  double* host_gaussian = new double[gaussian.size()];  // convert to double array
  for (int i = 0; i < gaussian.size(); i++) host_gaussian[i] = gaussian.at(i);
  // create buffer object
  gaussian_ = cl::Buffer(context_, CL_MEM_READ_ONLY, gaussian.size() * sizeof(double), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating gaussian kernel buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(gaussian_, CL_TRUE, 0, gaussian.size() * sizeof(double), static_cast<void*>(host_gaussian));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing gaussian kernel buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_gaussian;

  // gaussian size
  int* host_gaussian_size = new int[1];
  host_gaussian_size[0] = static_cast<int>(gaussian.size());
  // create buffer object
  gaussian_size_ = cl::Buffer(context_, CL_MEM_READ_ONLY, 1 * sizeof(int), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating gaussian size buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(gaussian_size_, CL_TRUE, 0, 1 * sizeof(int), static_cast<void*>(host_gaussian_size));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing gaussian size buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_gaussian_size;

  // scale amount
  int* host_scale = new int[1];
  host_scale[0] = static_cast<int>(motion_config_.scale_denominator);
  // create buffer object
  scale_ = cl::Buffer(context_, CL_MEM_READ_ONLY, 1 * sizeof(int), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating scale amount buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(scale_, CL_TRUE, 0, 1 * sizeof(int), static_cast<void*>(host_scale));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing scale amount buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_scale;

  // number of colors
  int* host_colors = new int[1];
  host_colors[0] = static_cast<int>(1);
  if (input_vid_.frame_format == DecompFrameFormat::kRGB) host_colors[0] = static_cast<int>(3);
  // create buffer object
  colors_ = cl::Buffer(context_, CL_MEM_READ_ONLY, 1 * sizeof(int), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating gaussian kernel buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(colors_, CL_TRUE, 0, 1 * sizeof(int), static_cast<void*>(host_colors));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing gaussian kernel buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_colors;

  // input frame
  unsigned char* host_input_frame = new unsigned char[input_frame_buffer_size_];
  for (int i = 0; i < input_frame_buffer_size_; i++) host_input_frame[i] = 0;  // initialize to zero
  // create buffer object
  input_frame_ = cl::Buffer(context_, CL_MEM_READ_ONLY, input_frame_buffer_size_ * sizeof(unsigned char), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating input frame buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(input_frame_, CL_TRUE, 0, input_frame_buffer_size_ * sizeof(unsigned char), static_cast<void*>(host_input_frame));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing input frame buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_input_frame;

  // input width
  int* host_input_width = new int[1];
  host_input_width[0] = static_cast<int>(input_vid_.width);
  // create buffer object
  input_width_ = cl::Buffer(context_, CL_MEM_READ_ONLY, 1 * sizeof(int), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating input width buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(input_width_, CL_TRUE, 0, 1 * sizeof(int), static_cast<void*>(host_input_width));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing input width buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_input_width;

  // scaled width
  int* host_scaled_width = new int[1];
  host_scaled_width[0] = static_cast<int>(scaled_width_);
  // create buffer object
  output_width_ = cl::Buffer(context_, CL_MEM_READ_ONLY, 1 * sizeof(int), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating scaled width buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(output_width_, CL_TRUE, 0, 1 * sizeof(int), static_cast<void*>(host_scaled_width));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing scaled width buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_scaled_width;

  // intermediate scaled frame
  unsigned char* host_intermediate = new unsigned char[intermediate_scaled_frame_buffer_size_];
  for (int i = 0; i < intermediate_scaled_frame_buffer_size_; i++) host_intermediate[i] = 0;  // initialize to zero
  // create buffer object
  intermediate_scaled_frame_ = cl::Buffer(context_, CL_MEM_READ_WRITE, intermediate_scaled_frame_buffer_size_ * sizeof(unsigned char), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating intermediate scaled frame buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(intermediate_scaled_frame_, CL_TRUE, 0, intermediate_scaled_frame_buffer_size_ * sizeof(unsigned char),
                                        static_cast<void*>(host_intermediate));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing intermediate scaled frame buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_intermediate;

  // scaled frame
  unsigned char* host_scaled = new unsigned char[scaled_frame_buffer_size_];
  for (int i = 0; i < scaled_frame_buffer_size_; i++) host_scaled[i] = 0;  // initialize to zero
  // create buffer object
  scaled_frame_ = cl::Buffer(context_, CL_MEM_READ_WRITE, scaled_frame_buffer_size_ * sizeof(unsigned char), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating scaled frame buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(scaled_frame_, CL_TRUE, 0, scaled_frame_buffer_size_ * sizeof(unsigned char), static_cast<void*>(host_scaled));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing scaled frame buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_scaled;
}

void MotionDetector::LoadBlurAndScaleKernels() {
  // Load vertical kernel
  int error = CL_SUCCESS;
  cl::Program vertical_program = LoadProgram(kBlurScaleVerticalFile);
  bs_vertical_kernel_ = cl::Kernel(vertical_program, "blur_and_scale_vertical", &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to create vertical blur and scale kernel with error code: " + std::to_string(error));

  // NOLINTBEGIN(readability-magic-numbers)
  // Set kernel args
  error = bs_vertical_kernel_.setArg(0, gaussian_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set vertical blur and scale kernel argument with error code: " + std::to_string(error));
  error = bs_vertical_kernel_.setArg(1, gaussian_size_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set vertical blur and scale kernel argument with error code: " + std::to_string(error));
  error = bs_vertical_kernel_.setArg(2, scale_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set vertical blur and scale kernel argument with error code: " + std::to_string(error));
  error = bs_vertical_kernel_.setArg(3, colors_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set vertical blur and scale kernel argument with error code: " + std::to_string(error));
  error = bs_vertical_kernel_.setArg(4, input_frame_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set vertical blur and scale kernel argument with error code: " + std::to_string(error));
  error = bs_vertical_kernel_.setArg(5, input_width_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set vertical blur and scale kernel argument with error code: " + std::to_string(error));
  error = bs_vertical_kernel_.setArg(6, intermediate_scaled_frame_);
  if (error != CL_SUCCESS) {
    throw std::runtime_error("Failed to set vertical blur and scale kernel intermediate scaled frame argument with error code: " + std::to_string(error));
  }

  // Load horizontal kernel
  cl::Program horizontal_program = LoadProgram(kBlurScaleHorizontalFile);
  bs_horizontal_kernel_ = cl::Kernel(horizontal_program, "blur_and_scale_horizontal", &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to create horizontal blur and scale kernel with error code: " + std::to_string(error));

  // Set kernel args
  error = bs_horizontal_kernel_.setArg(0, gaussian_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set vertical blur and scale kernel argument with error code: " + std::to_string(error));
  error = bs_horizontal_kernel_.setArg(1, gaussian_size_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set vertical blur and scale kernel argument with error code: " + std::to_string(error));
  error = bs_horizontal_kernel_.setArg(2, scale_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set vertical blur and scale kernel argument with error code: " + std::to_string(error));
  error = bs_horizontal_kernel_.setArg(3, intermediate_scaled_frame_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set vertical blur and scale kernel argument with error code: " + std::to_string(error));
  error = bs_horizontal_kernel_.setArg(4, input_width_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set vertical blur and scale kernel argument with error code: " + std::to_string(error));
  error = bs_horizontal_kernel_.setArg(5, output_width_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set vertical blur and scale kernel argument with error code: " + std::to_string(error));
  error = bs_horizontal_kernel_.setArg(6, scaled_frame_);
  if (error != CL_SUCCESS) {
    throw std::runtime_error("Failed to set vertical blur and scale kernel intermediate scaled frame argument with error code: " + std::to_string(error));
  }
  // NOLINTEND(readability-magic-numbers)
}

void MotionDetector::LoadStabilizeAndCompareBuffers() {
  // Fill frames vector with empty frames
  for (int i = 0; i < motion_config_.bg_stabil_length + motion_config_.motion_stabil_length + 1; i++) {
    frames_.push_back(new unsigned char[scaled_frame_buffer_size_]);
    for (int j = 0; j < scaled_frame_buffer_size_; j++) frames_.at(i)[j] = 0;  // initialize to 0
  }

  // Create buffers
  int error = CL_SUCCESS;
  // background frame to remove
  unsigned char* host_bg_to_remove = new unsigned char[scaled_frame_buffer_size_];
  for (int i = 0; i < scaled_frame_buffer_size_; i++) host_bg_to_remove[i] = 0;  // initialize to 0
  // create buffer object
  bg_frame_to_remove_ = cl::Buffer(context_, CL_MEM_READ_ONLY, scaled_frame_buffer_size_ * sizeof(unsigned char), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating background to remove buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(bg_frame_to_remove_, CL_TRUE, 0, scaled_frame_buffer_size_ * sizeof(unsigned char), static_cast<void*>(host_bg_to_remove));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing background to remove buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_bg_to_remove;

  // movement frame to remove
  unsigned char* host_mvt_to_remove = new unsigned char[scaled_frame_buffer_size_];
  for (int i = 0; i < scaled_frame_buffer_size_; i++) host_mvt_to_remove[i] = 0;  // initialize to 0
  // create buffer object
  mvt_frame_to_remove_ = cl::Buffer(context_, CL_MEM_READ_ONLY, scaled_frame_buffer_size_ * sizeof(unsigned char), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating movement to remove buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(mvt_frame_to_remove_, CL_TRUE, 0, scaled_frame_buffer_size_ * sizeof(unsigned char), static_cast<void*>(host_mvt_to_remove));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing movement to remove buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_mvt_to_remove;

  // background length
  double* host_bg_len = new double[1];
  host_bg_len[0] = static_cast<double>(motion_config_.bg_stabil_length);
  // create buffer object
  bg_length_ = cl::Buffer(context_, CL_MEM_READ_ONLY, 1 * sizeof(double), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating background length buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(bg_length_, CL_TRUE, 0, 1 * sizeof(double), static_cast<void*>(host_bg_len));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing background length buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_bg_len;

  // movement length
  double* host_mvt_len = new double[1];
  host_mvt_len[0] = static_cast<double>(motion_config_.motion_stabil_length);
  // create buffer object
  mvt_length_ = cl::Buffer(context_, CL_MEM_READ_ONLY, 1 * sizeof(double), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating movement length buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(mvt_length_, CL_TRUE, 0, 1 * sizeof(double), static_cast<void*>(host_mvt_len));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing movement length buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_mvt_len;

  // stabilized background frame
  double* host_bg = new double[scaled_frame_buffer_size_];
  for (int i = 0; i < scaled_frame_buffer_size_; i++) host_bg[i] = 0;  // initialize to 0
  // create buffer object
  stabilized_background_ = cl::Buffer(context_, CL_MEM_READ_WRITE, scaled_frame_buffer_size_ * sizeof(double), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating stabilized background buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(stabilized_background_, CL_TRUE, 0, scaled_frame_buffer_size_ * sizeof(double), static_cast<void*>(host_bg));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing stabilized background buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_bg;

  // stabilized movement frame
  double* host_mvt = new double[scaled_frame_buffer_size_];
  for (int i = 0; i < scaled_frame_buffer_size_; i++) host_mvt[i] = 0;  // initialize to 0
  // create buffer object
  stabilized_movement_ = cl::Buffer(context_, CL_MEM_READ_WRITE, scaled_frame_buffer_size_ * sizeof(double), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating stabilized movement buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(stabilized_movement_, CL_TRUE, 0, scaled_frame_buffer_size_ * sizeof(double), static_cast<void*>(host_mvt));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing stabilized movement buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_mvt;

  // pixel difference threshold
  unsigned int* host_pix_diff_thresh = new unsigned int[1];
  host_pix_diff_thresh[0] = motion_config_.min_pixel_diff;
  // create buffer object
  pixel_diff_threshold_ = cl::Buffer(context_, CL_MEM_READ_ONLY, 1 * sizeof(unsigned int), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating pixel difference threshold buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(pixel_diff_threshold_, CL_TRUE, 0, 1 * sizeof(unsigned int), static_cast<void*>(host_pix_diff_thresh));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing pixel difference threshold buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_pix_diff_thresh;

  // difference frame
  bool* host_diff = new bool[scaled_frame_buffer_size_];
  for (int i = 0; i < scaled_frame_buffer_size_; i++) host_diff[i] = false;  // initialize to false
  // create buffer object
  difference_frame_ = cl::Buffer(context_, CL_MEM_WRITE_ONLY, scaled_frame_buffer_size_ * sizeof(bool), nullptr, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Error creating stabilized movement buffer with error code: " + std::to_string(error));
  // write to OpenCL device
  error = cmd_queue_.enqueueWriteBuffer(difference_frame_, CL_TRUE, 0, scaled_frame_buffer_size_ * sizeof(bool), static_cast<void*>(host_diff));
  if (error != CL_SUCCESS) throw std::runtime_error("Error writing stabilized movement buffer with error code: " + std::to_string(error));
  // delete temp host memory
  delete[] host_diff;
}

void MotionDetector::LoadStabilizeAndCompareKernel() {
  // Load kernel
  int error = CL_SUCCESS;
  cl::Program stabilize_program = LoadProgram(kStabilizeFile);
  stabilize_kernel_ = cl::Kernel(stabilize_program, "stabilize_bg_mvt");
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to create stabilize background and movement kernel with error code: " + std::to_string(error));

  // NOLINTBEGIN(readability-magic-numbers)
  // Set kernel args
  error = stabilize_kernel_.setArg(0, bg_frame_to_remove_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set stabilize and compare frames kernel argument with error code: " + std::to_string(error));
  error = stabilize_kernel_.setArg(1, mvt_frame_to_remove_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set stabilize and compare frames kernel argument with error code: " + std::to_string(error));
  error = stabilize_kernel_.setArg(2, scaled_frame_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set stabilize and compare frames kernel argument with error code: " + std::to_string(error));
  error = stabilize_kernel_.setArg(3, bg_length_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set stabilize and compare frames kernel argument with error code: " + std::to_string(error));
  error = stabilize_kernel_.setArg(4, mvt_length_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set stabilize and compare frames kernel argument with error code: " + std::to_string(error));
  error = stabilize_kernel_.setArg(5, stabilized_background_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set stabilize and compare frames kernel argument with error code: " + std::to_string(error));
  error = stabilize_kernel_.setArg(6, stabilized_movement_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set stabilize and compare frames kernel argument with error code: " + std::to_string(error));
  error = stabilize_kernel_.setArg(7, pixel_diff_threshold_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set stabilize and compare frames kernel argument with error code: " + std::to_string(error));
  error = stabilize_kernel_.setArg(8, difference_frame_);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to set stabilize and compare frames kernel argument with error code: " + std::to_string(error));
  // NOLINTEND(readability-magic-numbers)

  bg_remove_loc_ = newest_frame_loc_ + 1;
  mvt_remove_loc_ = frames_.size() - motion_config_.motion_stabil_length;
}

cl::Program MotionDetector::LoadProgram(const std::string& filename) {
  // Read the program source
  std::ifstream ifs(filename);
  if (!ifs.good()) throw std::runtime_error("Error while opening OpenCL kernel file: " + filename);
  std::string source_code(std::istreambuf_iterator<char>(ifs), (std::istreambuf_iterator<char>()));

  // Create OpenCL program
  cl::Program::Sources source;
  source.push_back({source_code.c_str(), source_code.length()});
  int error = CL_SUCCESS;
  cl::Program program = cl::Program(context_, source, &error);
  if (error != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL program from kernel file: " + filename);

  // Build program and throw errors if fails
  error = program.build(OPEN_CL_COMPILE_FLAGS);
  if (error != CL_SUCCESS) {
    std::cout << "OpenCL build failed! Build Log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_) << std::endl;
    throw std::runtime_error("Failed to compile OpenCL kernel file: " + filename);
  }
  std::cout << "Successfully compiled OpenCL kernel file: " + filename << std::endl;

  return program;
}
