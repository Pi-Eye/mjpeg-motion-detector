// NOLINTBEGIN(readability-magic-numbers)
#include <catch2/catch_all.hpp>
#include <cmath>
#include <limits>

#define private public  // To test steps of motion detection
#include "motion_detector.hpp"

const int kDevice = 0;              // kSpecific device to run tests on
const int kErrorMarginAllowed = 3;  // How different pixel values are allowed to be due to rounding

TEST_CASE("Construct Detector") {
  SECTION("With Valid Input") {
    InputVideoSettings input_vid_set_sol = {640, 480, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {1, 1, 10, 5, 5, 0.5};
    DeviceConfig device_config_sol = {DeviceType::kGPU, 2};

    REQUIRE_NOTHROW(MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol));
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

TEST_CASE("Blur and Scale Step On RGB Frames") {
  // Smaller image
  PpmFile ppm0 = ReadPpm("../test-images/3x3-color-pixels-rgb.ppm");
  unsigned char* data0 = new unsigned char[ppm0.data.size()];
  for (int i = 0; i < ppm0.data.size(); i++) {
    data0[i] = ppm0.data.at(i);
  }
  // Larger image
  PpmFile ppm1 = ReadPpm("../test-images/9x9-color-pixels-rgb.ppm");
  unsigned char* data1 = new unsigned char[ppm1.data.size()];
  for (int i = 0; i < ppm1.data.size(); i++) {
    data1[i] = ppm1.data.at(i);
  }

  SECTION("With No Blur") {
    InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {0, 1, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);

    cl::Buffer blurred = motion_detector.BlurAndScale(data0);

    unsigned char* pixels = new unsigned char[3 * 3];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 3 * 3 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 255) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[1]) - 170) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[2]) - 170) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[3]) - 170) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[4]) - 85) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[5]) - 85) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[6]) - 85) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[7]) - 0) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[8]) - 255) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With 3x3 Blur") {
    InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {1, 1, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data0);

    unsigned char* pixels = new unsigned char[1];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 1 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 127) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With 1/2x Scale") {
    InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {0, 2, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data0);

    unsigned char* pixels = new unsigned char[1];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 1 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 170) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With 1/3x Scale") {
    InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {0, 3, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data0);

    unsigned char* pixels = new unsigned char[1];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 1 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 142) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With 1/2x Scale On Larger Image") {
    InputVideoSettings input_vid_set_sol = {9, 9, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {0, 2, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data1);

    unsigned char* pixels = new unsigned char[4 * 4];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 4 * 4 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 170) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[1]) - 170) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[2]) - 127) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[3]) - 170) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[4]) - 127) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[5]) - 191) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[6]) - 149) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[7]) - 127) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[8]) - 85) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[9]) - 149) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[10]) - 107) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[11]) - 85) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[12]) - 170) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[13]) - 170) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[14]) - 127) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[15]) - 170) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With 1/3x Scale On Larger Image") {
    InputVideoSettings input_vid_set_sol = {9, 9, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {0, 3, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data1);

    unsigned char* pixels = new unsigned char[3 * 3];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 3 * 3 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[1]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[2]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[3]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[4]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[5]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[6]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[7]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[8]) - 142) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With 1/2x Scale and 3x3 Blur On Larger Image") {
    InputVideoSettings input_vid_set_sol = {9, 9, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {1, 2, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data1);

    unsigned char* pixels = new unsigned char[2 * 2];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 2 * 2 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 150) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[1]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[2]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[3]) - 134) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With 1/3x Scale and 3x3 Blur On Larger Image") {
    InputVideoSettings input_vid_set_sol = {9, 9, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {1, 3, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data1);

    unsigned char* pixels = new unsigned char[1];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 1 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 142) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With No Scale and 3x3 Blur On Larger Image") {
    InputVideoSettings input_vid_set_sol = {9, 9, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {1, 1, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data1);

    unsigned char* pixels = new unsigned char[7 * 7];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 7 * 7 * sizeof(unsigned char), static_cast<void*>(pixels));

    std::vector<int> expected = {127, 139, 142, 127, 139, 142, 127, 124, 144, 139, 124, 144, 139, 124, 142, 154, 157, 142, 154, 157, 142, 127, 139, 142, 127,
                                 139, 142, 127, 124, 144, 139, 124, 144, 139, 124, 142, 154, 157, 142, 154, 157, 142, 127, 139, 142, 127, 139, 142, 127};
    for (int i = 0; i < 49; i++) {
      REQUIRE(abs(static_cast<int>(pixels[i]) - expected.at(i)) < kErrorMarginAllowed);
    }

    delete[] pixels;
  }

  delete[] data0;
  delete[] data1;
}

TEST_CASE("Blur and Scale Step On Grayscale Frames") {
  // Smaller image
  PpmFile ppm0 = ReadPpm("../test-images/3x3-color-pixels-grayscale.ppm");
  unsigned char* data0 = new unsigned char[ppm0.data.size()];
  for (int i = 0; i < ppm0.data.size(); i++) {
    data0[i] = ppm0.data.at(i);
  }
  // Larger image
  PpmFile ppm1 = ReadPpm("../test-images/9x9-color-pixels-grayscale.ppm");
  unsigned char* data1 = new unsigned char[ppm1.data.size()];
  for (int i = 0; i < ppm1.data.size(); i++) {
    data1[i] = ppm1.data.at(i);
  }

  SECTION("With No Blur") {
    InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kGray};
    MotionConfig motion_config_sol = {0, 1, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);

    cl::Buffer blurred = motion_detector.BlurAndScale(data0);

    unsigned char* pixels = new unsigned char[3 * 3];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 3 * 3 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 255) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[1]) - 227) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[2]) - 105) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[3]) - 179) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[4]) - 77) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[5]) - 150) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[6]) - 28) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[7]) - 0) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[8]) - 255) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With 3x3 Blur") {
    InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kGray};
    MotionConfig motion_config_sol = {1, 1, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data0);

    unsigned char* pixels = new unsigned char[1];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 1 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 133) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With 1/2x Scale") {
    InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kGray};
    MotionConfig motion_config_sol = {0, 2, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data0);

    unsigned char* pixels = new unsigned char[1];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 1 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 185) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With 1/3x Scale") {
    InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kGray};
    MotionConfig motion_config_sol = {0, 3, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data0);

    unsigned char* pixels = new unsigned char[1];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 1 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 142) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With 1/2x Scale On Larger Image") {
    InputVideoSettings input_vid_set_sol = {9, 9, DecompFrameFormat::kGray};
    MotionConfig motion_config_sol = {0, 2, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data1);

    unsigned char* pixels = new unsigned char[4 * 4];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 4 * 4 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 185) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[1]) - 172) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[2]) - 140) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[3]) - 185) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[4]) - 128) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[5]) - 161) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[6]) - 147) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[7]) - 128) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[8]) - 71) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[9]) - 153) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[10]) - 120) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[11]) - 71) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[12]) - 185) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[13]) - 172) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[14]) - 140) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[15]) - 185) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With 1/3x Scale On Larger Image") {
    InputVideoSettings input_vid_set_sol = {9, 9, DecompFrameFormat::kGray};
    MotionConfig motion_config_sol = {0, 3, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data1);

    unsigned char* pixels = new unsigned char[3 * 3];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 3 * 3 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[1]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[2]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[3]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[4]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[5]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[6]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[7]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[8]) - 142) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With 1/2x Scale and 3x3 Blur On Larger Image") {
    InputVideoSettings input_vid_set_sol = {9, 9, DecompFrameFormat::kGray};
    MotionConfig motion_config_sol = {1, 2, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data1);

    unsigned char* pixels = new unsigned char[2 * 2];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 2 * 2 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 146) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[1]) - 142) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[2]) - 141) < kErrorMarginAllowed);
    REQUIRE(abs(static_cast<int>(pixels[3]) - 136) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With 1/3x Scale and 3x3 Blur On Larger Image") {
    InputVideoSettings input_vid_set_sol = {9, 9, DecompFrameFormat::kGray};
    MotionConfig motion_config_sol = {1, 3, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data1);

    unsigned char* pixels = new unsigned char[1];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 1 * sizeof(unsigned char), static_cast<void*>(pixels));

    REQUIRE(abs(static_cast<int>(pixels[0]) - 142) < kErrorMarginAllowed);
    delete[] pixels;
  }

  SECTION("With No Scale and 3x3 Blur On Larger Image") {
    InputVideoSettings input_vid_set_sol = {9, 9, DecompFrameFormat::kGray};
    MotionConfig motion_config_sol = {1, 1, 10, 2, 0, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    cl::Buffer blurred = motion_detector.BlurAndScale(data1);

    unsigned char* pixels = new unsigned char[7 * 7];
    motion_detector.cmd_queue_.enqueueReadBuffer(blurred, CL_TRUE, 0, 7 * 7 * sizeof(unsigned char), static_cast<void*>(pixels));

    std::vector<int> expected = {132, 144, 143, 132, 144, 143, 132, 124, 142, 132, 124, 142, 132, 124, 145, 151, 154, 145, 151, 154, 145, 132, 144, 143, 132,
                                 144, 143, 132, 124, 142, 132, 124, 142, 132, 124, 145, 151, 154, 145, 151, 154, 145, 132, 144, 143, 132, 144, 143, 132};
    for (int i = 0; i < 49; i++) {
      REQUIRE(abs(static_cast<int>(pixels[i]) - expected.at(i)) < kErrorMarginAllowed);
    }

    delete[] pixels;
  }

  delete[] data0;
  delete[] data1;
}

TEST_CASE("Stabilize And Compare Frames Step") {
  // Fully white frame
  unsigned char* data0 = new unsigned char[3 * 3 * 3];
  for (int i = 0; i < 27; i++) {
    data0[i] = 255;
  }
  // Fully black frame
  unsigned char* data1 = new unsigned char[3 * 3 * 3];
  for (int i = 0; i < 27; i++) {
    data1[i] = 0;
  }
  // Fully grey frame
  unsigned char* data2 = new unsigned char[3 * 3 * 3];
  for (int i = 0; i < 27; i++) {
    data2[i] = 127;
  }

  SECTION("Compare Difference On The Same Frames") {
    InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {0, 1, 1, 1, kErrorMarginAllowed, 0.0};
    DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

    MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);

    // Fill with 2 white frames
    motion_detector.BlurAndScale(data0);
    motion_detector.StabilizeAndCompareFrames();
    motion_detector.BlurAndScale(data0);

    cl::Buffer& difference_frame = motion_detector.StabilizeAndCompareFrames();

    bool* differences = new bool[3 * 3];
    motion_detector.cmd_queue_.enqueueReadBuffer(difference_frame, CL_TRUE, 0, 3 * 3 * sizeof(bool), static_cast<void*>(differences));

    // Should be no differences
    for (int i = 0; i < 9; i++) REQUIRE(differences[i] == false);
    delete[] differences;
  }

  SECTION("Compare Difference On The Different Frames") {
    {
      InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kRGB};
      MotionConfig motion_config_sol = {0, 1, 1, 1, (127 - kErrorMarginAllowed), 0.0};  // Check that difference is above the expected value (127) within margin
      DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

      MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);

      // Fill with 1 black and 1 grey frame
      motion_detector.BlurAndScale(data1);
      motion_detector.StabilizeAndCompareFrames();
      motion_detector.BlurAndScale(data2);

      cl::Buffer& difference_frame = motion_detector.StabilizeAndCompareFrames();

      bool* differences = new bool[3 * 3];
      motion_detector.cmd_queue_.enqueueReadBuffer(difference_frame, CL_TRUE, 0, 3 * 3 * sizeof(bool), static_cast<void*>(differences));

      // Should have differences
      for (int i = 0; i < 9; i++) REQUIRE(differences[i] == true);
      delete[] differences;
    }

    {  // Do the same thing but with a higher threshold so there should be no change
      InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kRGB};
      MotionConfig motion_config_sol = {0, 1, 1, 1, (127 + kErrorMarginAllowed), 0.0};  // Check that difference is above the expected value (127) within margin
      DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

      MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);

      // Fill with 1 black and 1 grey frame
      motion_detector.BlurAndScale(data1);
      motion_detector.StabilizeAndCompareFrames();
      motion_detector.BlurAndScale(data2);

      cl::Buffer& difference_frame = motion_detector.StabilizeAndCompareFrames();

      bool* differences = new bool[3 * 3];
      motion_detector.cmd_queue_.enqueueReadBuffer(difference_frame, CL_TRUE, 0, 3 * 3 * sizeof(bool), static_cast<void*>(differences));

      // Should have differences
      for (int i = 0; i < 9; i++) REQUIRE(differences[i] == false);
      delete[] differences;
    }
  }

  SECTION("Compare Difference On The Different Frames While Averaging Background") {
    {
      InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kRGB};
      MotionConfig motion_config_sol = {0, 1, 10, 1, (25 - kErrorMarginAllowed), 0.0};  // Check that difference is above the expected value (25) within margin
      DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

      MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);

      // Fill with 1 black and 10 white frames
      motion_detector.BlurAndScale(data1);
      for (int i = 0; i < 10; i++) {
        motion_detector.StabilizeAndCompareFrames();
        motion_detector.BlurAndScale(data0);
      }

      cl::Buffer& difference_frame = motion_detector.StabilizeAndCompareFrames();

      bool* differences = new bool[3 * 3];
      motion_detector.cmd_queue_.enqueueReadBuffer(difference_frame, CL_TRUE, 0, 3 * 3 * sizeof(bool), static_cast<void*>(differences));

      // Should have differences
      for (int i = 0; i < 9; i++) REQUIRE(differences[i] == true);
      delete[] differences;
    }

    {  // Do the same thing but with a higher threshold so there should be no change
      InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kRGB};
      MotionConfig motion_config_sol = {0, 1, 10, 1, (25 + kErrorMarginAllowed), 0.0};  // Check that difference is above the expected value (25) within margin
      DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

      MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);

      // Fill with 1 black and 10 white frames
      motion_detector.BlurAndScale(data1);
      for (int i = 0; i < 10; i++) {
        motion_detector.StabilizeAndCompareFrames();
        motion_detector.BlurAndScale(data0);
      }

      cl::Buffer& difference_frame = motion_detector.StabilizeAndCompareFrames();

      bool* differences = new bool[3 * 3];
      motion_detector.cmd_queue_.enqueueReadBuffer(difference_frame, CL_TRUE, 0, 3 * 3 * sizeof(bool), static_cast<void*>(differences));

      // Should have differences
      for (int i = 0; i < 9; i++) REQUIRE(differences[i] == false);
      delete[] differences;
    }
  }

  SECTION("Compare Difference On The Different Frames While Averaging Movement") {
    {
      InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kRGB};
      MotionConfig motion_config_sol = {0, 1, 1, 10, (25 - kErrorMarginAllowed), 0.0};  // Check that difference is above the expected value (25) within margin
      DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

      MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);

      // Fill with 10 black and 1 white frame
      for (int i = 0; i < 10; i++) {
        motion_detector.BlurAndScale(data1);
        motion_detector.StabilizeAndCompareFrames();
      }
      motion_detector.BlurAndScale(data0);

      cl::Buffer& difference_frame = motion_detector.StabilizeAndCompareFrames();

      bool* differences = new bool[3 * 3];
      motion_detector.cmd_queue_.enqueueReadBuffer(difference_frame, CL_TRUE, 0, 3 * 3 * sizeof(bool), static_cast<void*>(differences));

      // Should have differences
      for (int i = 0; i < 9; i++) REQUIRE(differences[i] == true);
      delete[] differences;
    }

    {  // Do the same thing but with a higher threshold so there should be no change
      InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kRGB};
      MotionConfig motion_config_sol = {0, 1, 1, 10, (25 + kErrorMarginAllowed), 0.0};  // Check that difference is above the expected value (25) within margin
      DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

      MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);

      // Fill with 10 black and 1 white frame
      for (int i = 0; i < 10; i++) {
        motion_detector.BlurAndScale(data1);
        motion_detector.StabilizeAndCompareFrames();
      }
      motion_detector.BlurAndScale(data0);

      cl::Buffer& difference_frame = motion_detector.StabilizeAndCompareFrames();

      bool* differences = new bool[3 * 3];
      motion_detector.cmd_queue_.enqueueReadBuffer(difference_frame, CL_TRUE, 0, 3 * 3 * sizeof(bool), static_cast<void*>(differences));

      // Should have differences
      for (int i = 0; i < 9; i++) REQUIRE(differences[i] == false);
      delete[] differences;
    }
  }

  delete[] data0;
  delete[] data1;
  delete[] data2;
}

TEST_CASE("Detect On Frame") {
  // Fully white frame
  unsigned char* data0 = new unsigned char[3 * 3];
  for (int i = 0; i < 9; i++) {
    data0[i] = 255;
  }
  // Fully half white half black frame
  unsigned char* data1 = new unsigned char[3 * 3];
  for (int i = 0; i < 5; i++) {
    data1[i] = 0;
  }
  for (int i = 5; i < 9; i++) {
    data1[i] = 255;
  }

  SECTION("With Half Changed Frame") {
    {
      InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kGray};
      MotionConfig motion_config_sol = {0, 1, 1, 1, 5, 0.5};
      DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

      MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);

      motion_detector.DetectOnFrame(data0);
      bool motion = motion_detector.DetectOnFrame(data1);

      REQUIRE(motion == true);
    }

    {  // Same thing again with difference threshold to ensure threshold is checked
      InputVideoSettings input_vid_set_sol = {3, 3, DecompFrameFormat::kGray};
      MotionConfig motion_config_sol = {0, 1, 1, 1, 5, 0.6};
      DeviceConfig device_config_sol = {DeviceType::kSpecific, kDevice};

      MotionDetector motion_detector = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);

      motion_detector.DetectOnFrame(data0);
      bool motion = motion_detector.DetectOnFrame(data1);

      REQUIRE(motion == false);
    }
  }

  delete[] data0;
  delete[] data1;
}
// NOLINTEND(readability-magic-numbers)