// NOLINTBEGIN(readability-magic-numbers)
#include <catch2/catch_all.hpp>
#include <cmath>
#include <limits>

#include "motion_detector.hpp"

TEST_CASE("Construct Detector") {
  SECTION("With Valid Input") {
    InputVideoSettings input_vid_set_sol = {640, 480, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {1, 1, 10, 5, 5, 0.5};
    DeviceConfig device_config_sol = {DeviceType::kGPU, 2};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);

    InputVideoSettings input_vid_set = motion_detector.GetInputVideoSettings();
    MotionConfig motion_config = motion_detector.GetMotionConfig();
    DeviceConfig device_config = motion_detector.GetDeviceConfig();

    REQUIRE(input_vid_set.width == input_vid_set_sol.width);
    REQUIRE(input_vid_set.height == input_vid_set_sol.height);
    REQUIRE(input_vid_set.frame_format == input_vid_set_sol.frame_format);

    REQUIRE(motion_config.gaussian_size == motion_config.gaussian_size);
    REQUIRE(motion_config.scale_denominator == motion_config.scale_denominator);
    REQUIRE(motion_config.bg_stabil_length == motion_config.bg_stabil_length);
    REQUIRE(motion_config.motion_stabil_length == motion_config.motion_stabil_length);
    REQUIRE(motion_config.min_pixel_diff == motion_config.min_pixel_diff);
    REQUIRE(motion_config.min_changed_pixels == motion_config.min_changed_pixels);

    REQUIRE(device_config.device_type == device_config.device_type);
    REQUIRE(device_config.device_choice == device_config.device_choice);
  }

  SECTION("With Invalid Input") {
    // Invalid Width
    InputVideoSettings input_vid_set_sol = {0, 480, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {1, 1, 10, 5, 0, 0.5};
    DeviceConfig device_config_sol = {DeviceType::kGPU, 2};

    REQUIRE_THROWS(MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol));

    // Invalid Height
    input_vid_set_sol = {640, 0, DecompFrameFormat::kRGB};
    motion_config_sol = {1, 1, 10, 5, 0, 0.5};
    device_config_sol = {DeviceType::kGPU, 2};

    REQUIRE_THROWS(MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol));

    // Invalid Scale Denominator
    input_vid_set_sol = {640, 480, DecompFrameFormat::kRGB};
    motion_config_sol = {1, 0, 10, 5, 0, 0.5};
    device_config_sol = {DeviceType::kGPU, 2};

    REQUIRE_THROWS(MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol));

    // Invalid Background Stabilization Length
    input_vid_set_sol = {640, 480, DecompFrameFormat::kRGB};
    motion_config_sol = {1, 1, 0, 5, 0, 0.5};
    device_config_sol = {DeviceType::kGPU, 2};

    REQUIRE_THROWS(MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol));

    // Invalid Movement Stabilization Length
    input_vid_set_sol = {640, 480, DecompFrameFormat::kRGB};
    motion_config_sol = {1, 1, 10, 0, 0, 0.5};
    device_config_sol = {DeviceType::kGPU, 2};

    REQUIRE_THROWS(MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol));

    // Invalid Minimun Changed Pixels
    input_vid_set_sol = {640, 480, DecompFrameFormat::kRGB};
    motion_config_sol = {1, 1, 10, 5, 0, -0.5};
    device_config_sol = {DeviceType::kGPU, 2};

    REQUIRE_THROWS(MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol));

    // Invalid Minimun Changed Pixels
    input_vid_set_sol = {640, 480, DecompFrameFormat::kRGB};
    motion_config_sol = {1, 1, 10, 5, 0, 1.1};
    device_config_sol = {DeviceType::kGPU, 2};

    REQUIRE_THROWS(MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol));

    // Invalid Gaussian Size and Scale Denominator Combination
    input_vid_set_sol = {3, 3, DecompFrameFormat::kRGB};
    motion_config_sol = {1, 2, 10, 5, 0, 0.1};
    device_config_sol = {DeviceType::kGPU, 2};

    REQUIRE_THROWS(MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol));
  }
}

TEST_CASE("Detect Motion On RGB Frames") {  // NOLINT(readability-function-cognitive-complexity)
  std::string file_path = "../test-images/3x3-color-pixels-rgb.ppm";
  PpmFile ppm0 = ReadPpm(file_path);
  unsigned char* data = new unsigned char[ppm0.data.size()];
  for (int i = 0; i < ppm0.data.size(); i++) {
    data[i] = ppm0.data.at(i);
  }

  SECTION("Blur and Scale Step") {
    InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {1, 1, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kGPU, 0};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data);

    cl::CommandQueue cl_queue = motion_detector.GetCLQueue();
    int* pixel = new int[3];
    cl_queue.enqueueReadBuffer(blurred, static_cast<cl_bool>(true), 0, 3 * sizeof(int), reinterpret_cast<void*>(pixel));
    REQUIRE(abs(pixel[0] - (280.5 / 16)) < 0.001);
    REQUIRE(abs(pixel[1] - (204 / 16)) < 0.001);
    REQUIRE(abs(pixel[2] - (153 / 16)) < 0.001);
  }

  SECTION("Stabilize Frames Step") {
    InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {0, 1, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kGPU, 0};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);

    for (int i = 0; i < 20; i++) {
      motion_detector.BlurAndScale(data);
    }

    StabilizedFrames stabilized_frames = motion_detector.StabilizeFrames();

    cl::CommandQueue cl_queue = motion_detector.GetCLQueue();
    int* frame = new int[27];
    cl_queue.enqueueReadBuffer(stabilized_frames.stabilized_bg, static_cast<cl_bool>(true), 0, static_cast<size_t>(27 * sizeof(int)), reinterpret_cast<void*>(frame));
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        int index = 3 * (row * 3 + col);
        REQUIRE(abs(frame[index] - (int)data[index]) < 0.001);
        REQUIRE(abs(frame[index + 1] - (int)data[index + 1]) < 0.001);
        REQUIRE(abs(frame[index + 2] - (int)data[index + 2]) < 0.001);
      }
    }
    cl_queue.enqueueReadBuffer(stabilized_frames.stabilized_mvt, static_cast<cl_bool>(true), 0, static_cast<size_t>(27 * sizeof(int)), reinterpret_cast<void*>(frame));
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        int index = 3 * (row * 3 + col);
        REQUIRE(abs(frame[index] - (int)data[index]) < 0.001);
        REQUIRE(abs(frame[index + 1] - (int)data[index + 1]) < 0.001);
        REQUIRE(abs(frame[index + 2] - (int)data[index + 2]) < 0.001);
      }
    }
  }

  SECTION("Compare Frames Step") {}
}
// NOLINTEND(readability-magic-numbers)
