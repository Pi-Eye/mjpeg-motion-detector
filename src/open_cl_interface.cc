#include "open_cl_interface.hpp"

OpenCLInterface::OpenCLInterface(DeviceConfig device_config) { CreateContext(device_config); }

void OpenCLInterface::CreateContext(DeviceConfig device_config) {}

std::vector<cl::Device> OpenCLInterface::ListDevices(cl_device_type device_type) {}