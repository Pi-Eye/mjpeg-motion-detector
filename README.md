# MJPEG Motion Detector

## About

A simple motion detector for MJPEG streams.

### Built With

* C++
* OpenCL
* libjpeg-turbo

## Getting Started

### Prerequisites

1. An OpenCL 1.2 compatible device ([compatability list](https://www.khronos.org/conformance/adopters/conformant-products/opencl))

2. [vcpkg](https://vcpkg.io/en/index.html)

3. [Cmake](https://cmake.org/)

### Building

1. Clone the repository
    ```sh
    git clone https://github.com/Pi-Eye/mjpeg-motion-detector.git
    ```

2. Install dependencies
    ```sh
    vcpkg install opencl libjpeg-turbo
    ```

    note. also install `catch2` if you would like to run tests and benchmarks

3. Configure project

    In the directory you just cloned, run:
    ```sh
    cmake -DMAKE_TOOLCHAIN_FILE:STRING="path/to/vcpkg/scripts/buildsystems/vcpkg.cmake" . -B./build
    ```

4. Build

    Main
    ```sh
    cmake --build ./build --target exec
    ```
    Benchmarks
    ```sh
    cmake --build ./build --target benchmark
    ```
    Tests
    ```sh
    cmake --build ./build --target test
    ```

5. Run
  
    Compiled executables will be found in the `/bin` directory

## Usage

### Example Motion Detector

```cpp
#include <iostream>
#include <ostream>
#include <vector>

#include "motion_detector.hpp"

int main() {
  unsigned int width = 1920;
  unsigned int height = 1080;                                       // width and height of incoming jpeg frames (acceptable values: > 0)
  DecompFrameFormat frame_format = DecompFrameFormat::kRGB;         // format to decompress frames into (acceptable values: kRGB, kGray)

  unsigned int gaussian_radius = 2;                                 // radius of gaussian blur (0 means no blur) (acceptable values: > 0)
  unsigned int scale = 2;                                           // amount to scale frame down by to save on computation (acceptable values: > 0)
  unsigned int bg_stabilization_len = 10;                           // number of frames to average to form background frame (acceptable values: > 0)
  unsigned int mvt_stabilization_len = 2;                           // number of frames to average to form movement frame (acceptable values: > 0)
  unsigned int min_pixel_diff = 5;                                  // amount pixels need to be different by to be considered different (acceptable values: >= 0)
  unsigned int min_changed_pixels = 0.2;                            // percentage of pixels that need to be different to be considered motion (acceptable values: 0.0 - 1.0)
  DecompFrameMethod decomp_method = DecompFrameMethod::kAccurate;   // algorithm to use to decompress jpeg (acceptable values: kAccurate, kFast)

  DeviceType device_type = DeviceType::kSpecific;                   // type of OpenCL device to select (acceptable values: kCPU, kGPU, kSpecific)
  int device_choice = 0;                                            // index of OpenCL device to use (acceptable values: >=0)

  InputVideoSettings input_video_settings = {width, height, frame_format};
  MotionConfig motion_config = {gaussian_radius, scale, bg_stabilization_len, mvt_stabilization_len, min_pixel_diff, min_changed_pixels};
  DeviceConfig device_config = {device_type, device_choice};
  std::ostream* info_stream = &std::cout;                           // pipe info stream of motion detector to cout

  MotionDetector motion = MotionDetector(input_video_settings, motion_config, device_config, info_stream);

  std::vector<std::pair<unsigned char*, unsigned long>> frames;     // some group of jpeg frames to detect motion on 
                                                                    // (unsigned char* - frame, unsigned long - size of jpeg frame buffer)

  for (int i = 0; i < frames.size(); i++) {
    bool has_motion = motion.DetectOnFrame(frames.at(i).first, frames.at(i).second);      // Detect if there is motion on given frame

    if (has_motion) {
      std::cout << "Detected motion on frame number: " << i << std::endl;
    }
  }
}

```


## License

Distributed uner the GPL-3.0 License. See `LICENSE.txt` for more information.

## Contact

Bennett Wu - bwu1324@gmail.com