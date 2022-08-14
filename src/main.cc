#include <iostream>

//#include "motion_detector.hpp"
//#include "generate_gaussian.hpp"
//#include "jpeg_decompressor.hpp"
#include "open_cl_interface.hpp"

int main() {
  try {
    std::cout << "Hello, World" << std::endl;

    OpenCLInterface open_cl = OpenCLInterface({DeviceType::kSpecific, 1});

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