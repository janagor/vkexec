#ifndef VKEXEC_DEVICE_PROCS_HPP
#define VKEXEC_DEVICE_PROCS_HPP

#include <vulkan/vulkan.h>

namespace vkexec {

/// Device entry points loaded via `vkGetDeviceProcAddr`. Null when unavailable.
struct device_procs
{
  PFN_vkGetBufferDeviceAddress get_buffer_device_address{};

  PFN_vkWriteResourceDescriptorsEXT write_resource_descriptors{};
  PFN_vkWriteSamplerDescriptorsEXT write_sampler_descriptors{};
  PFN_vkCmdBindResourceHeapEXT cmd_bind_resource_heap{};
  PFN_vkCmdBindSamplerHeapEXT cmd_bind_sampler_heap{};
  PFN_vkCmdPushDataEXT cmd_push_data{};
};

}// namespace vkexec

#endif// VKEXEC_DEVICE_PROCS_HPP
