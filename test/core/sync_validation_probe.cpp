#include "recording_buffer.hpp"
#include <atomic>
#include <cstdio>
#include <string_view>
#include <vkexec/submit_scope.hpp>
#include <vkexec/sync_wait.hpp>

namespace {
auto VKAPI_CALL observe(VkDebugUtilsMessageSeverityFlagBitsEXT,
  VkDebugUtilsMessageTypeFlagsEXT,
  VkDebugUtilsMessengerCallbackDataEXT const *data,
  void *user) -> VkBool32
{
  if (data && data->pMessageIdName
      && std::string_view(data->pMessageIdName).find("SYNC-HAZARD") != std::string_view::npos) {
    static_cast<std::atomic<bool> *>(user)->store(true);
  }
  return VK_FALSE;
}
struct messenger
{
  VkInstance instance{};
  VkDebugUtilsMessengerEXT handle{};
  PFN_vkDestroyDebugUtilsMessengerEXT destroy{};
  ~messenger()
  {
    if (handle) { destroy(instance, handle, nullptr); }
  }
};
}// namespace

// This is an opt-in positive-control probe, not a valid graph integration test.
// It deliberately records a hazard, observes validation, then discards the buffer.
auto main() -> int
{
  vkexec::scheduler_options options{};
  options.validation_layers = true;
  auto made = vkexec::try_sync_wait_value(vkexec::factory::make_context(options));
  if (!made) {
    std::fputs("Validation context unavailable\n", stderr);
    return 1;
  }
  auto ctx = std::move(*made);
  auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
    vkGetInstanceProcAddr(ctx->instance(), "vkCreateDebugUtilsMessengerEXT"));
  auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
    vkGetInstanceProcAddr(ctx->instance(), "vkDestroyDebugUtilsMessengerEXT"));
  if (!create || !destroy) { return 1; }
  std::atomic<bool> detected{};
  messenger debug{ ctx->instance(), VK_NULL_HANDLE, destroy };
  VkDebugUtilsMessengerCreateInfoEXT info{};
  info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
  info.pfnUserCallback = observe;
  info.pUserData = &detected;
  if (create(ctx->instance(), &info, nullptr, &debug.handle) != VK_SUCCESS) { return 1; }
  auto allocated = vkexec::test::recording_buffer::make(*ctx);
  if (!allocated) { return 1; }
  auto buffer = std::move(*allocated);
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  if (!opened) { return 1; }
  auto scope = vkexec::expected_take(opened);
  vkCmdFillBuffer(scope.cmd, buffer->buffer, 0, buffer->byte_size, 1);
  vkCmdFillBuffer(scope.cmd, buffer->buffer, 0, buffer->byte_size, 2);
  if (vkResetCommandBuffer(scope.cmd, 0) != VK_SUCCESS) { return 1; }
  if (!detected.load()) {
    std::fputs("Synchronization validation was NOT observed\n", stderr);
    return 1;
  }
  std::puts("SYNCVAL_PROBE_CONFIRMED: intentional unsubmitted WAW hazard detected");
  return 0;
}
