#ifndef MOTION_DETECTOR_HPP
#define MOTION_DETECTOR_HPP

#include <CL/cl.hpp>
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
 */
struct MotionConfig {
  unsigned int gaussian_size;
  unsigned int scale_denominator;
  unsigned int bg_stabil_length;
  unsigned int motion_stabil_length;
  unsigned int min_pixel_diff;
  float min_changed_pixels;
};

/**
 * StabilizedFrames - The pair of stabilized frames
 *
 * stabilized_bg:   averaged background frames
 * stabilized_mvt:  averaged movement frames
 */
struct StabilizedFrames {
  cl::Buffer stabilized_bg;
  cl::Buffer stabilized_mvt;
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
  MotionDetector(InputVideoSettings input_vid_settings, MotionConfig motion_config, DeviceConfig device_config);

  /**
   * ~DetectMotion() - Deconstructor for DetectMotion
   */
  ~MotionDetector();

  /**
   * DetectOnFrame() - Processes a frame for motion detection
   *
   * frame:     image in the format used to construct DetectMotion
   *              (note: image format will not be checked)
   * returns:   bool - if motion is detected or not
   */
  bool DetectOnFrame(unsigned char* frame);

  /**
   * BlurAndScale() - Blurs and scales an image using selected gaussian size
   *
   * image:     image to be blurred and scaled
   * returns:   cl::Buffer - blurred image
   */
  cl::Buffer BlurAndScale(unsigned char* image);

  /**
   * StabilizeFrames() - Averages background and motion frames
   *
   * returns:   StabilizedFrames - pair of stabilized frames
   */
  StabilizedFrames StabilizeFrames();

  /**
   * GetInputVideoSettings() - Gets input video settings
   *
   * returns:   InputVideoSettings - input video settings of motion detector
   */
  InputVideoSettings GetInputVideoSettings() const;

  /**
   * GetMotionConfig() - Gets motion detection configuration
   *
   * returns:   MotionConfig - motion detection configuration of motion detector
   */
  MotionConfig GetMotionConfig() const;

  /**
   * GetDeviceConfig() - Gets device configuration
   *
   * returns:   DeviceConfig - processing device configuration of motion detector
   */
  DeviceConfig GetDeviceConfig() const;

  /**
   * GetCLQueue() - Gets OpenCL Command Queue
   *
   * returns:   cl::CommandQueue - OpenCL Command Queue of motion detector
   */
  cl::CommandQueue& GetCLQueue();

 private:
  OpenCLInterface opencl_;          // OpenCL interface
  std::vector<cl::Buffer> frames_;  // Vector of OpenCL buffers for frames
  cl::Buffer gaussian_kernel_;      // OpenCL buffer of gaussian kernel

  InputVideoSettings input_vid_;  // Metadata about MJPEG stream coming in
  MotionConfig motion_config_;    // Settings for how exactly to run motion detection
  DeviceConfig device_config_;    // Settings for which device to run motion detection on
};

#endif