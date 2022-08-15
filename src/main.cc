#include <iostream>

#include "generate_gaussian.hpp"
#include "jpeg_decompressor.hpp"
#include "motion_detector.hpp"
#include "open_cl_interface.hpp"

int main() {
  try {
    std::cout << "Hello, World" << std::endl;

    OpenCLInterface open_cl = OpenCLInterface({DeviceType::kSpecific, 1});

    InputVideoSettings input_vid_set_sol = {640, 480, DecompFrameFormat::kRGB};
    MotionConfig motion_config_sol = {1, 1, 10, 5, 5, 0.5};
    DeviceConfig device_config_sol = {DeviceType::kGPU, 2};
    MotionDetector motion = MotionDetector(input_vid_set_sol, motion_config_sol, device_config_sol);
    // Catch and print all execptions
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return -1;
  } catch (const std::string& ex) {
    std::cerr << ex << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "Unknown Exception" << std::endl;
    return -1;
  }
}