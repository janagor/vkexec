#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PROCS_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PROCS_HPP

#include <vkexec/context.hpp>

#include <vulkan/vulkan.h>

namespace vkexec {

struct descriptor_heap_procs
{
  PFN_vkWriteResourceDescriptorsEXT write_resource_descriptors{};
  PFN_vkCmdBindResourceHeapEXT cmd_bind_resource_heap{};
  PFN_vkCmdPushDataEXT cmd_push_data{};
};

[[nodiscard]] auto load_descriptor_heap_procs(VkDevice device) -> descriptor_heap_procs;

[[nodiscard]] auto descriptor_heap_procs_for(context const &ctx) -> descriptor_heap_procs const &;

[[nodiscard]] auto descriptor_heap_available(descriptor_heap_procs const &procs) -> bool;

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PROCS_HPP
