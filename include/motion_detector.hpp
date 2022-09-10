#ifndef MOTION_DETECTOR_HPP
#define MOTION_DETECTOR_HPP

#include <CL/opencl.h>

#include <CL/cl2.hpp>
#include <ostream>
#include <vector>

#include "jpeg_decompressor.hpp"
#include "open_cl_interface.hpp"

/**
 * InputVideoSettings - Metadata of decompressed video stream
 *
 * width:         width of video in pixels
 * height:        height of video in pixels
 * video_format:  video format
 */
struct InputVideoSettings {
  unsigned int width;
  unsigned int height;
  DecompFrameFormat frame_format;
};

/**
 * MotionConfig - Configuration for motion detection
 *
 * gaussian_size:         size of gaussian blur (0 mean no blur, 1 means 3x3, 2 means 5x5, etc.)
 * scale_denominator:     amount to scale down input by
 * bg_stabil_length:      number of frames to average to form stabilized background
 * motion_stabil_length:  number of frames to average to form stabilized motion
 * min_pixel_diff:        minimum difference between pixels to count as different
 * min_changed_pixels:    minimum pecentage of pixels that need to change in a frame to count as a different frame
 * decomp_method:         decompression method to use for jpeg
 */
struct MotionConfig {
  unsigned int gaussian_size;
  unsigned int scale_denominator;
  unsigned int bg_stabil_length;
  unsigned int motion_stabil_length;
  unsigned int min_pixel_diff;
  float min_changed_pixels;
  DecompFrameMethod decomp_method;
};

/**
 * MotionDetector - Detects motion on MJPEG stream
 */
class MotionDetector {
 public:
  /**
   * DetectMotion() - Constructor for DetectMotion
   *
   * input_vid_settings:   Metadata about MJPEG stream coming in
   * motion_config:        Settings for how exactly to run motion detection
   * device_config:        Settings for which device to run motion detection on
   */
  MotionDetector(InputVideoSettings input_vid_settings, MotionConfig motion_config, DeviceConfig device_config, std::ostream* output);

  /**
   * ~DetectMotion() - Deconstructor for DetectMotion
   */
  ~MotionDetector();

  /**
   * DetectOnFrame() - Processes a MJPEG frame for motion detection
   *
   * frame:     JPEG image
   * size:      Size of JPEG image buffer
   * returns:   bool - if motion is detected or not
   */
  bool DetectOnFrame(unsigned char* frame, unsigned long size);

  /**
   * DetectOnDecompressedFrame() - Processes a decompressed frame for motion detection
   *
   * frame:     image in the format used to construct DetectMotion
   *              (note: image format will not be checked)
   * returns:   bool - if motion is detected or not
   */
  bool DetectOnDecompressedFrame(unsigned char* frame);

  /**
   * BlurAndScale() - Blurs and scales an image using selected gaussian size
   *
   * image:     image to be blurred and scaled
   * returns:   cl::Buffer& - blurred and scaled image
   */
  cl::Buffer& BlurAndScale(unsigned char* frame);

  /**
   * StabilizeAndCompareFrames() - Averages background and motion frames and compares them
   *
   * returns:   cl::Buffer& - image of differences
   */
  cl::Buffer& StabilizeAndCompareFrames();

 private:
  /**
   * ValidateSettings() - Validates settings for motion detector
   */
  void ValidateSettings() const;

  /**
   * InitOpenCL() - Sets up OpenCL opbjects
   */
  void InitOpenCL();

  /**
   * InitWorkSizes() - Calculates and creates OpenCL device work sizes
   */
  void InitWorkSizes();

  /**
   * CalculateBufferSizes() - Calculates sizes of buffers needed for motion detection
   */
  void CalculateBufferSizes();

  /**
   * LoadBlurAndScaleBuffers() - Loads OpenCL buffers for blurring and scaling frame
   */
  void LoadBlurAndScaleBuffers();

  /**
   * LoadBlurAndScaleKernels() - Loads OpenCL kernels for blurring and scaling frame
   */
  void LoadBlurAndScaleKernels();

  /**
   * LoadStabilizeAndCompareBuffer() - Loads OpenCL buffers for stabilizing background and movement and comparing them
   */
  void LoadStabilizeAndCompareBuffers();

  /**
   * LoadStabilizeAndCompareKernel() - Loads OpenCL kernels for stabilizing background and movement and comparing them
   */
  void LoadStabilizeAndCompareKernel();

  /**
   * LoadProgram() - Loads OpenCL program from given filename
   *
   * returns:  cl::Program - built OpenCL program
   */
  cl::Program LoadProgram(const std::string& filename);

  JpegDecompressor decompressor_;  // Jpeg decompressor

  cl::Device device_;           // OpenCL device motion detection will run on
  cl::Context context_;         // OpenCL context for device
  cl::CommandQueue cmd_queue_;  // OpenCL command queue for device

  // Inputs
  cl::Buffer gaussian_;       // OpenCL buffer of gaussian kernel
  cl::Buffer gaussian_size_;  // OpenCL buffer of gaussian kernel size
  cl::Buffer scale_;          // OpenCL buffer of scale factor
  cl::Buffer colors_;         // OpenCL buffer of number of colors
  cl::Buffer input_width_;    // OpenCL buffer of width of input frame
  cl::Buffer output_width_;   // OpenCL buffer of width of scaled frame
  cl::Buffer input_frame_;    // OpenCL buffer for incoming frame to be processed

  // Kernel
  cl::Kernel bs_vertical_kernel_;  // OpenCL kernel for blurring and scaling frame vertically
  // Output
  cl::Buffer intermediate_scaled_frame_;  // OpenCL buffer for scaled frame after just vertical scaling

  // Kernel
  cl::Kernel bs_horizontal_kernel_;  // OpenCL kernel for blurring and scaling frame horizontally
  // Output
  cl::Buffer scaled_frame_;  // OpenCL buffer for scaled frame

  // Inputs
  cl::Buffer bg_length_;             // OpenCL buffer for length of background
  cl::Buffer mvt_length_;            // OpenCL buffer for length of movement
  cl::Buffer bg_frame_to_remove_;    // OpenCL buffer for background frame to be removed from average
  cl::Buffer mvt_frame_to_remove_;   // OpenCL buffer for movement frame to be removed from average
  cl::Buffer pixel_diff_threshold_;  // OpenCL buffer for amount pixel needs to be different by to be different
  // Kernel
  cl::Kernel stabilize_kernel_;  // OpenCL kernel for stabilizing background and forground
  // Outputs
  cl::Buffer stabilized_background_;  // OpenCL buffer for stabilzed background
  cl::Buffer stabilized_movement_;    // OpenCL buffer for stabilzied movement
  cl::Buffer difference_frame_;       // OpenCL buffer for difference between background and movement

  cl::NDRange scaled_global_work_size_2d_;               // 2D Work size of fully scaled down frame
  cl::NDRange intermediate_scaled_global_work_size_2d_;  // 2D Work size of vertically scaled down frame
  cl::NDRange motion_thread_block_size_2d_;              // 2D Work size of thread for motion detection
  cl::NDRange scaled_global_work_size_1d_;               // 1D Work size of fully scaled down frame
  cl::NDRange motion_thread_block_size_1d_;              // 1D Work size of thread for motion detection

  unsigned int newest_frame_loc_ = 0;   // Index of the newest frame in the list of all frames
  unsigned int bg_remove_loc_;          // Index of background frame to remove in the list of all frames
  unsigned int mvt_remove_loc_;         // Index of movement frame to remove in the list of all frames
  std::vector<unsigned char*> frames_;  // List of all frames

  unsigned int diff_threshold_;  // Number of pixels that need to be different for the frame to be counted as motion

  unsigned int input_frame_buffer_size_;                // Size of frame input
  unsigned int intermediate_scaled_frame_buffer_size_;  // Size of intermediate scaling step
  unsigned int scaled_frame_buffer_size_;               // Size of scaled frame for motion detection (no color data)
  unsigned int scaled_width_;                           // Width of scaled frames
  unsigned int scaled_height_;                          // Height of scaled frames

  InputVideoSettings input_vid_;  // Metadata about MJPEG stream coming in
  MotionConfig motion_config_;    // Settings for how exactly to run motion detection
  DeviceConfig device_config_;    // Settings for which device to run motion detection on

  std::ostream* info;  // Output stream for info messages
};

#endif
