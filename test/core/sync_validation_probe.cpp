#include "recording_buffer.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vkexec/context.hpp>
#include <vkexec/result.hpp>
#include <vkexec/submit_scope.hpp>
#include <vkexec/sync_wait.hpp>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>

namespace {
constexpr std::uint32_t k_first_fill_value = 1;
constexpr std::uint32_t k_second_fill_value = 2;

auto VKAPI_CALL observe(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
  VkDebugUtilsMessageTypeFlagsEXT message_type,
  VkDebugUtilsMessengerCallbackDataEXT const *data,
  void *user) -> VkBool32
{
  (void)severity;
  (void)message_type;
  if (data != nullptr && data->pMessageIdName != nullptr
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
  messenger(VkInstance value, PFN_vkDestroyDebugUtilsMessengerEXT destroy_function)
    : instance(value), destroy(destroy_function)
  {}
  messenger(messenger const &) = delete;
  auto operator=(messenger const &) -> messenger & = delete;
  messenger(messenger &&) = delete;
  auto operator=(messenger &&) -> messenger & = delete;
  ~messenger()
  {
    if (handle != VK_NULL_HANDLE) { destroy(instance, handle, nullptr); }
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
    (void)std::fputs("Validation context unavailable\n", stderr);
    return EXIT_FAILURE;
  }
  auto ctx = std::move(*made);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- Vulkan entry point loading requires a function pointer cast
  auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
    vkGetInstanceProcAddr(ctx->instance(), "vkCreateDebugUtilsMessengerEXT"));
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- Vulkan entry point loading requires a function pointer cast
  auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
    vkGetInstanceProcAddr(ctx->instance(), "vkDestroyDebugUtilsMessengerEXT"));
  if (create == nullptr || destroy == nullptr) { return EXIT_FAILURE; }
  std::atomic<bool> detected{};
  messenger debug{ ctx->instance(), destroy };
  VkDebugUtilsMessengerCreateInfoEXT info{};
  info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
  info.pfnUserCallback = observe;
  info.pUserData = &detected;
  if (create(ctx->instance(), &info, nullptr, &debug.handle) != VK_SUCCESS) { return EXIT_FAILURE; }
  auto allocated = vkexec::test::recording_buffer::make(*ctx);
  if (!allocated) { return EXIT_FAILURE; }
  auto buffer = std::move(*allocated);
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  if (!opened) { return EXIT_FAILURE; }
  auto scope = vkexec::expected_take(opened);
  vkCmdFillBuffer(scope.cmd, buffer->buffer, 0, buffer->k_byte_size, k_first_fill_value);
  vkCmdFillBuffer(scope.cmd, buffer->buffer, 0, buffer->k_byte_size, k_second_fill_value);
  if (vkResetCommandBuffer(scope.cmd, 0) != VK_SUCCESS) { return EXIT_FAILURE; }
  if (!detected.load()) {
    (void)std::fputs("Synchronization validation was NOT observed\n", stderr);
    return EXIT_FAILURE;
  }
  (void)std::puts("SYNCVAL_PROBE_CONFIRMED: intentional unsubmitted WAW hazard detected");
  return EXIT_SUCCESS;
}
