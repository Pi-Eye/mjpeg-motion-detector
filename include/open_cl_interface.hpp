#ifndef OPEN_CL_INTERFACE_HPP
#define OPEN_CL_INTERFACE_HPP

#include <CL/opencl.h>

#include <CL/cl2.hpp>

/**
 * DeviceType - Selector for how to select OpenCL device to run motion detection on
 *
 * kCPU:      Select first CPU device
 * kGPU:      Select first GPU device
 * kSpecific: Select a specific device ID
 */
enum class DeviceType { kCPU, kGPU, kSpecific };

/**
 * DeviceConfig - Selector for which OpenCL device to run motion detection on
 *
 * device_type:   how to select device
 * device_choice: device id
 */
struct DeviceConfig {
  DeviceType device_type;
  int device_choice;
};

/**
 * OpenCLInterface - Class with useful methods for interfacing with OpenCL
 */
class OpenCLInterface {
 public:
  /**
   * ListDevices() - Gets list of avaliable OpenCL devices
   *
   * device_type:   OpenCL device type to show
   * returns:       std::vector<cl::Device> - list of all avaliable devices where each item is the OpenCL device and the index is it's unique id
   */
  static std::vector<cl::Device> ListDevices(cl_device_type device_type);

  /**
   * CreateContext() - Selects OpenCL device based on configuration
   *
   * device_config:   Settings for which device to select
   */
  static cl::Device GetDevice(DeviceConfig device_config);
};

#endif
