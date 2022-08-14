#include "open_cl_interface.hpp"

#include <iostream>
#include <vector>

std::vector<cl::Device> OpenCLInterface::ListDevices(cl_device_type device_type) {
  std::vector<cl::Device> devices;
  // Get all OpenCL platforms
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  for (int i = 0; i < platforms.size(); i++) {
    // Get all OpenCL devices on given platform
    std::vector<cl::Device> platform_devices;
    platforms.at(i).getDevices(device_type, &platform_devices);

    // Add devices on platform to list of all devices
    for (int j = 0; j < platform_devices.size(); j++) {
      devices.push_back(platform_devices.at(j));
    }
  }

  return devices;
}

OpenCLInterface::OpenCLInterface(DeviceConfig device_config) {
  CreateContext(device_config);
  cmd_queue_ = cl::CommandQueue(context_, device_);
}

void OpenCLInterface::CreateContext(DeviceConfig device_config) {
  // Select device and throw error if not found
  switch (device_config.device_type) {
    case (DeviceType::kCPU):
    default: {
      std::vector<cl::Device> avaliable_devices = ListDevices(CL_DEVICE_TYPE_CPU);
      if (avaliable_devices.empty()) throw std::runtime_error("No OpenCL compatable CPU's found");
      device_ = avaliable_devices.at(0);
      break;
    }
    case (DeviceType::kGPU): {
      std::vector<cl::Device> avaliable_devices = ListDevices(CL_DEVICE_TYPE_GPU);
      if (avaliable_devices.empty()) throw std::runtime_error("No OpenCL compatable GPU's found");
      device_ = avaliable_devices.at(0);
      break;
    }
    case (DeviceType::kSpecific): {
      std::vector<cl::Device> avaliable_devices = ListDevices(CL_DEVICE_TYPE_ALL);
      if (avaliable_devices.size() <= device_config.device_choice) throw std::runtime_error("Selected OpenCL device was not avaliable");
      device_ = avaliable_devices.at(device_config.device_choice);
      break;
    }
  }
  std::cout << "Selected device: " + device_.getInfo<CL_DEVICE_NAME>() << std::endl;

  // Create OpenCL context and command queue
  context_ = cl::Context(device_);
}

cl::Context& OpenCLInterface::GetContext() { return context_; }

cl::Device& OpenCLInterface::GetDevice() { return device_; }

cl::CommandQueue& OpenCLInterface::GetCommandQueue() { return cmd_queue_; }