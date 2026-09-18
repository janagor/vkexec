#include <vkexec_extensions/descriptor_heap/strategy.hpp>

#include <vkexec_extensions/descriptor_heap/push_data.hpp>

#include <vkexec/context.hpp>
#include <vkexec/detail/result.hpp>
#include <vkexec/error.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <span>

namespace vkexec::detail {

auto heap_descriptor_backend::push_bytes(context const *ctx,
  VkCommandBuffer cmd,
  VkPipelineBindPoint /*bind_point*/,
  std::span<std::byte const> bytes) -> status
{
  if (bytes.empty()) { return {}; }
  if (ctx == nullptr) { return fail(errc::invalid_argument, "heap descriptor push requires a context"); }
  return cmd_push_data(*ctx, cmd, bytes);
}

}// namespace vkexec::detail
