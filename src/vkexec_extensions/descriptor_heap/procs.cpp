#include <vkexec_extensions/descriptor_heap/procs.hpp>

#include <vkexec/context.hpp>
#include <vkexec/device_proc.hpp>
#include <vkexec_extensions/common/proc_cache.hpp>

#include <vulkan/vulkan_core.h>

namespace vkexec {

auto load_descriptor_heap_procs(VkDevice device) -> descriptor_heap_procs
{
  descriptor_heap_procs procs{};
  procs.write_resource_descriptors =
    resolve_device_proc<PFN_vkWriteResourceDescriptorsEXT>(device, "vkWriteResourceDescriptorsEXT");
  procs.cmd_bind_resource_heap =
    resolve_device_proc<PFN_vkCmdBindResourceHeapEXT>(device, "vkCmdBindResourceHeapEXT");
  procs.cmd_push_data = resolve_device_proc<PFN_vkCmdPushDataEXT>(device, "vkCmdPushDataEXT");
  return procs;
}

auto descriptor_heap_procs_for(context const &ctx) -> descriptor_heap_procs const &
{ return ext::detail::cached_procs<descriptor_heap_procs>(ctx.device(), load_descriptor_heap_procs); }

auto descriptor_heap_available(descriptor_heap_procs const &procs) -> bool
{
  return procs.write_resource_descriptors != nullptr && procs.cmd_bind_resource_heap != nullptr
         && procs.cmd_push_data != nullptr;
}

}// namespace vkexec
