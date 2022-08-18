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

cl::Device OpenCLInterface::GetDevice(DeviceConfig device_config) {
  // Select device and throw error if not found
  cl::Device device;
  switch (device_config.device_type) {
    case (DeviceType::kCPU):
    default: {
      std::vector<cl::Device> avaliable_devices = ListDevices(CL_DEVICE_TYPE_CPU);
      if (avaliable_devices.empty()) throw std::runtime_error("No OpenCL compatable CPU's found");
      device = avaliable_devices.at(0);
      break;
    }
    case (DeviceType::kGPU): {
      std::vector<cl::Device> avaliable_devices = ListDevices(CL_DEVICE_TYPE_GPU);
      if (avaliable_devices.empty()) throw std::runtime_error("No OpenCL compatable GPU's found");
      device = avaliable_devices.at(0);
      break;
    }
    case (DeviceType::kSpecific): {
      std::vector<cl::Device> avaliable_devices = ListDevices(CL_DEVICE_TYPE_ALL);
      if (avaliable_devices.size() <= device_config.device_choice) throw std::runtime_error("Selected OpenCL device was not avaliable");
      device = avaliable_devices.at(device_config.device_choice);
      break;
    }
  }
  return device;
}
