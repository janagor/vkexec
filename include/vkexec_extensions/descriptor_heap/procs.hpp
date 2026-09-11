#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PROCS_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PROCS_HPP

//! \file
//! Device entry points for `VK_EXT_descriptor_heap`.

#include <vkexec/context.hpp>

#include <vulkan/vulkan.h>

namespace vkexec {

/**
 * Function pointers for descriptor-heap host writes, heap bind, and push-data.
 *
 * Null members mean the extension entry point was not resolved.
 *
 * @see load_descriptor_heap_procs, descriptor_heap_procs_for
 */
struct descriptor_heap_procs
{
  PFN_vkWriteResourceDescriptorsEXT write_resource_descriptors{};
  PFN_vkCmdBindResourceHeapEXT cmd_bind_resource_heap{};
  PFN_vkCmdPushDataEXT cmd_push_data{};
};

//! Loads descriptor-heap PFNs for `device` (may return null members).
[[nodiscard]] auto load_descriptor_heap_procs(VkDevice device) -> descriptor_heap_procs;

//! Returns a cached proc table for `ctx`'s device.
[[nodiscard]] auto descriptor_heap_procs_for(context const &ctx) -> descriptor_heap_procs const &;

//! True when all required descriptor-heap PFNs are non-null.
[[nodiscard]] auto descriptor_heap_available(descriptor_heap_procs const &procs) -> bool;

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_PROCS_HPP
