
#include <catch2/catch_all.hpp>
#include <fstream>
#include <iostream>
#include <ostream>

#include "motion_detector.hpp"

const int kDevice = 0;  // OpenCL device to run tests on

std::ostream* empty_output = new std::ostream(0);

struct Configs {
  MotionConfig motion;
  InputVideoSettings video;
};

// NOLINTBEGIN(readability-*)
std::vector<std::pair<unsigned int, unsigned int>> resolutions = {{640, 480}, {1280, 720}, {1920, 1080}};
std::vector<DecompFrameFormat> frame_formats = {DecompFrameFormat::kRGB};
std::vector<DecompFrameMethod> decomp_methods = {DecompFrameMethod::kAccurate};
std::vector<unsigned int> gaussian_sizes = {0, 1, 2};
std::vector<unsigned int> scale_denominators = {10, 5, 1};

std::vector<Configs> PermutateConfigs() {
  std::vector<Configs> configs;
  for (int a = 0; a < decomp_methods.size(); a++) {
    for (int b = 0; b < frame_formats.size(); b++) {
      for (int c = 0; c < gaussian_sizes.size(); c++) {
        for (int d = 0; d < scale_denominators.size(); d++) {
          for (int e = 0; e < resolutions.size(); e++) {
            configs.push_back({
                {gaussian_sizes.at(c), scale_denominators.at(d), 10, 2, 1, 0.1, decomp_methods.at(a)},
                {resolutions.at(e).first, resolutions.at(e).second, frame_formats.at(b)},
            });
          }
        }
      }
    }
  }
  return configs;
}
// NOLINTEND(readability-*)

struct JpegFile {
  unsigned long filesize;
  unsigned char* data;
};

JpegFile ReadJpeg(const std::string& path) {
  // Create filestream and throw error if it fails
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.good()) throw std::runtime_error("Error reading file: " + path);

  // Get filesize
  ifs.seekg(0, std::ios::end);
  unsigned int filesize = ifs.tellg();
  ifs.seekg(0);

  // Create destination for file
  unsigned char* data = new unsigned char[filesize];

  // Read file
  ifs.read(reinterpret_cast<char*>(data), filesize);

  return {filesize, data};
}

TEST_CASE("Benchmark Motion Detection") {
  std::vector<Configs> configs = PermutateConfigs();

  for (int i = 0; i < configs.size(); i++) {
    std::string name;
    {  // Create name of benchmark
      name.append(std::to_string(configs.at(i).video.width));
      name.append("x");
      name.append(std::to_string(configs.at(i).video.height));
      if (configs.at(i).video.frame_format == DecompFrameFormat::kGray) name.append(" (Grayscale)");
      if (configs.at(i).video.frame_format == DecompFrameFormat::kRGB) name.append(" (RGB)");

      if (configs.at(i).motion.decomp_method == DecompFrameMethod::kAccurate) name.append("   (Accurate)");
      if (configs.at(i).motion.decomp_method == DecompFrameMethod::kFast) name.append("   (Fast)");
      name.append("\n");

      name.append("Gaussian Size: ");
      name.append(std::to_string(configs.at(i).motion.gaussian_size));
      name.append("   Scale: ");
      name.append(std::to_string(configs.at(i).motion.scale_denominator));
      name.append("\n");
    }

    MotionDetector motion = MotionDetector(configs.at(i).video, configs.at(i).motion, {DeviceType::kSpecific, kDevice}, empty_output);

    // Get relavent test image
    // NOLINTBEGIN(readability-magic-numbers)
    std::string jpg_name;

    if (configs.at(i).video.width == 640) jpg_name = "../test-images/640x480-test-image.jpg";
    if (configs.at(i).video.width == 1280) jpg_name = "../test-images/1280x720-test-image.jpg";
    if (configs.at(i).video.width == 1920) jpg_name = "../test-images/1920x1080-test-image.jpg";

    JpegFile jpeg_frame = ReadJpeg(jpg_name);
    // NOLINTEND(readability-magic-numbers)

    BENCHMARK(std::string(name)) { return motion.DetectOnFrame(jpeg_frame.data, jpeg_frame.filesize); };

    delete[] jpeg_frame.data;
  }
}