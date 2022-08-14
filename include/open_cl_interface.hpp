#ifndef OPEN_CL_INTERFACE_HPP
#define OPEN_CL_INTERFACE_HPP

#include <CL/cl.hpp>

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
   * OpenCLInterface() - Constructor for OpenCLInterface
   *
   * device_config:   Settings for which device to select
   */
  explicit OpenCLInterface(DeviceConfig device_config);

  /**
   * CreateContext() - Selects OpenCL device based on configuration and creates context
   *
   * device_config:   Settings for which device to select
   */
  void CreateContext(DeviceConfig device_config);

  /**
   * GetContext() - Get OpenCL context
   *
   * returns:   cl::Context& - reference of OpenCL context
   */
  cl::Context& GetContext();

  /**
   * GetDevice() - Get OpenCL device
   *
   * returns:   cl::Device& - reference of OpenCL device
   */
  cl::Device& GetDevice();

  /**
   * GetCommandQueue() - Get OpenCL command queue
   *
   * returns:   cl::CommandQueue& - reference of OpenCL command queue
   */
  cl::CommandQueue& GetCommandQueue();

 private:
  cl::Context context_;         // OpenCL context
  cl::Device device_;           // OpenCL device being used
  cl::CommandQueue cmd_queue_;  // OpenCL command queue
};

#endif