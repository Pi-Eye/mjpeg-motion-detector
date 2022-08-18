// NOLINTBEGIN
#include <CL/cl.hpp>
#include <iostream>
#include <string>

#include "generate_gaussian.hpp"
#include "jpeg_decompressor.hpp"
#include "motion_detector.hpp"
#include "open_cl_interface.hpp"

#define THREAD_BLOCK_SIZE 1

int main() {
  try {
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
// NOLINTEND