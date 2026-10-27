#include <VkBootstrap.h>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/synchronization.hpp>
#include <vkexec/device_capabilities.hpp>
#include <vkexec/device_procs.hpp>
#include <vkexec/error.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vulkan/vulkan_core.h>

#include "detail/completion_waiter.hpp"
#include "detail/host_agent.hpp"
#include "detail/vk_bootstrap_context_helpers.hpp"
#include "vkexec/vulkan_requirements.hpp"

namespace vkexec {

context::context(factory_access /*access*/, [[maybe_unused]] uninitialized_tag tag)
  : impl_(context_bootstrap::make_context_state())
{}

auto detail::make_context_factory::operator()() const -> result<std::unique_ptr<::vkexec::context>>
{
  // Factory may allocate; sender::start() catches and maps to set_error.
  // NOLINTNEXTLINE(bugprone-exception-escape)
  auto ctx =
    std::make_unique<::vkexec::context>(::vkexec::context::factory_access{}, ::vkexec::context::uninitialized_tag{});

  VKEXEC_TRY(ctx->init_headless(opts));
  return ctx;
}

auto detail::adopt_context_factory::operator()() const -> result<std::unique_ptr<::vkexec::context>>
{
  // Factory may allocate; sender::start() catches and maps to set_error.
  // NOLINTNEXTLINE(bugprone-exception-escape)
  auto ctx =
    std::make_unique<::vkexec::context>(::vkexec::context::factory_access{}, ::vkexec::context::uninitialized_tag{});

  VKEXEC_TRY(ctx->init_adopted(info));
  return ctx;
}

auto context::instance() const noexcept -> VkInstance { return impl_->instance.instance; }
auto context::physical_device() const noexcept -> VkPhysicalDevice { return impl_->physical_device.physical_device; }
auto context::device() const noexcept -> VkDevice { return impl_->device.device; }
auto context::compute_queue() const noexcept -> VkQueue { return impl_->compute_queue; }
auto context::graphics_queue() const noexcept -> VkQueue { return impl_->graphics_queue; }
auto context::present_queue() const noexcept -> VkQueue { return impl_->present_queue; }
auto context::queue_family() const noexcept -> std::uint32_t { return impl_->queue_family; }
auto context::graphics_queue_family() const noexcept -> std::uint32_t { return impl_->graphics_family; }
auto context::present_queue_family() const noexcept -> std::uint32_t { return impl_->present_family; }
auto context::compute_queue_ref() const noexcept -> queue_ref
{ return queue_ref{ .queue = compute_queue(), .family = queue_family() }; }
auto context::graphics_queue_ref() const noexcept -> queue_ref
{ return queue_ref{ .queue = graphics_queue(), .family = graphics_queue_family() }; }
auto context::present_queue_ref() const noexcept -> queue_ref
{ return queue_ref{ .queue = present_queue(), .family = present_queue_family() }; }
auto context::owns_instance() const noexcept -> bool { return impl_->owns_instance; }
auto context::owns_device() const noexcept -> bool { return impl_->owns_device; }
auto context::presentation_enabled() const noexcept -> bool { return impl_->presentation_enabled; }
auto context::capabilities() const noexcept -> device_capabilities const & { return impl_->capabilities; }
auto context::api_version() const noexcept -> std::uint32_t { return impl_->capabilities.api_version; }
auto context::procs() const noexcept -> device_procs const & { return impl_->procs; }

auto context::extension_procs_entry(std::string_view key, std::function<std::shared_ptr<void>()> const &load) const
  -> std::shared_ptr<void> const &
{
  std::scoped_lock const lock(impl_->extension_procs_mutex);
  auto const found = impl_->extension_procs.find(key);
  if (found != impl_->extension_procs.end()) { return found->second; }
  return impl_->extension_procs.emplace(std::string{ key }, load()).first->second;
}

auto detail::synchronization_backend_for(context const &ctx) noexcept -> synchronization_backend
{ return ctx.impl_->synchronization; }

auto context::init_common_resources() -> status
{
  // Shared path for create() and window surface completion after the device exists.
  VKEXEC_TRY(load_device_procs());
  VKEXEC_TRY(impl_->register_known_queues());
  impl_->completion_waiter = std::make_unique<detail::completion_waiter>(
    impl_->device.device, impl_->compute_queue, impl_->queue_synchronization.state(impl_->compute_queue));
  impl_->host_agent = std::make_unique<detail::host_agent>();
  return {};
}

auto context::init_headless(scheduler_options const &opts) -> status
{
  impl_->creation_requirements = opts.requirements;
  impl_->capabilities.api_version = context_bootstrap::resolve_api_version(impl_->creation_requirements);
  context_bootstrap::configure_optional_synchronization2(impl_->creation_requirements, impl_->capabilities.api_version);

  scheduler_options effective_opts = opts;
  effective_opts.requirements = impl_->creation_requirements;
  VKEXEC_TRY_ASSIGN(built_instance, context_bootstrap::build_headless_instance(effective_opts, api_version()));
  impl_->instance = built_instance;
  impl_->has_instance = true;
  impl_->owns_instance = true;

  VKEXEC_TRY_ASSIGN(selected_physical,
    context_bootstrap::select_physical_device(
      impl_->instance, impl_->creation_requirements, api_version(), VK_NULL_HANDLE, false, impl_->capabilities));
  impl_->physical_device = std::move(selected_physical);
  impl_->synchronization = detail::select_synchronization_backend(impl_->capabilities);


  VKEXEC_TRY(context_bootstrap::build_device_into(impl_->physical_device, impl_->device));
  impl_->has_device = true;
  impl_->owns_device = true;

  VKEXEC_TRY(fetch_queues(false));
  return init_common_resources();
}

auto context::init_adopted(context_adopt_info const &info) -> status
{
  // Borrowed handles: never set owns_* for instance/device.
  if (info.device == VK_NULL_HANDLE) { return fail(errc::invalid_argument, "context::adopt requires a VkDevice"); }
  if (info.compute_queue == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "context::adopt requires a compute VkQueue");
  }

  impl_->capabilities = info.capabilities;
  impl_->synchronization = detail::select_synchronization_backend(impl_->capabilities);
  impl_->instance.instance = info.instance;
  impl_->physical_device.physical_device = info.physical_device;
  impl_->device.device = info.device;
  impl_->has_instance = info.instance != VK_NULL_HANDLE;
  impl_->has_device = true;

  impl_->compute_queue = info.compute_queue;
  impl_->queue_family = info.compute_queue_family;

  if (info.graphics_queue != VK_NULL_HANDLE) {
    impl_->graphics_queue = info.graphics_queue;
    impl_->graphics_family = info.graphics_queue_family;
  } else {
    // Fall back to compute so graphics helpers can still query a queue.
    impl_->graphics_queue = impl_->compute_queue;
    impl_->graphics_family = impl_->queue_family;
  }

  if (info.present_queue != VK_NULL_HANDLE) {
    impl_->present_queue = info.present_queue;
    impl_->present_family = info.present_queue_family;
  } else {
    impl_->present_queue = impl_->graphics_queue;
    impl_->present_family = impl_->graphics_family;
  }

  VKEXEC_TRY(load_device_procs());

  VKEXEC_TRY(impl_->register_known_queues());
  impl_->completion_waiter = std::make_unique<detail::completion_waiter>(
    impl_->device.device, impl_->compute_queue, impl_->queue_synchronization.state(impl_->compute_queue));
  impl_->host_agent = std::make_unique<detail::host_agent>();
  return {};
}

context::context(factory_access /*access*/,
  instance_only_tag tag,
  scheduler_options const &opts,
  std::vector<char const *> const &instance_extensions)
  : impl_(context_bootstrap::make_context_state())
{
  (void)tag;
  impl_->creation_requirements = opts.requirements;
  impl_->capabilities.api_version = context_bootstrap::resolve_api_version(impl_->creation_requirements);
  context_bootstrap::configure_optional_synchronization2(impl_->creation_requirements, api_version());
  scheduler_options effective_opts = opts;
  effective_opts.requirements = impl_->creation_requirements;
  auto built_instance =
    context_bootstrap::build_instance_with_extensions(effective_opts, api_version(), instance_extensions);
  if (!built_instance) { detail::contract_violation("context instance-only construction failed"); }
  impl_->instance = expected_take(built_instance);
  impl_->has_instance = true;
  impl_->owns_instance = true;
}

auto context::complete_for_surface(VkSurfaceKHR surface) -> status
{
  // Second-phase init used by window: instance already exists; select a present-capable device.
  if (surface == VK_NULL_HANDLE) { return fail(errc::invalid_argument, "complete_for_surface requires a surface"); }

  VKEXEC_TRY_ASSIGN(selected_physical,
    context_bootstrap::select_physical_device(
      impl_->instance, impl_->creation_requirements, api_version(), surface, true, impl_->capabilities));
  impl_->physical_device = std::move(selected_physical);
  impl_->synchronization = detail::select_synchronization_backend(impl_->capabilities);


  VKEXEC_TRY(context_bootstrap::build_device_into(impl_->physical_device, impl_->device));
  impl_->has_device = true;
  impl_->owns_device = true;

  VKEXEC_TRY(fetch_queues(true));
  VKEXEC_TRY(init_common_resources());
  impl_->presentation_enabled = true;
  return {};
}

auto context::fetch_queues(bool want_present) -> status
{
  // Prefer dedicated compute; otherwise share the graphics queue (common on mobile / iGPUs).
  if (auto graphics = impl_->device.get_queue_and_index(vkb::QueueType::graphics)) {
    impl_->graphics_queue = graphics->first;
    impl_->graphics_family = graphics->second;
  }

  if (auto compute = impl_->device.get_queue_and_index(vkb::QueueType::compute)) {
    impl_->compute_queue = compute->first;
    impl_->queue_family = compute->second;
  } else if (impl_->graphics_queue != VK_NULL_HANDLE) {
    impl_->compute_queue = impl_->graphics_queue;
    impl_->queue_family = impl_->graphics_family;
  } else {
    return fail(errc::unsupported, "no compute or graphics queue available");
  }

  if (impl_->graphics_queue == VK_NULL_HANDLE) {
    impl_->graphics_queue = impl_->compute_queue;
    impl_->graphics_family = impl_->queue_family;
  }

  if (want_present) {
    auto present = impl_->device.get_queue_and_index(vkb::QueueType::present);
    if (!present) { return fail(errc::unsupported, "no present queue available"); }
    impl_->present_queue = present->first;
    impl_->present_family = present->second;
  } else {
    impl_->present_queue = impl_->graphics_queue;
    impl_->present_family = impl_->graphics_family;
  }
  return {};
}

auto context::load_device_procs() -> status
{
  impl_->procs = {};
  if (impl_->device.device == VK_NULL_HANDLE) { return fail(errc::invalid_argument, "device is null"); }

  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  impl_->procs.get_buffer_device_address = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
    vkGetDeviceProcAddr(impl_->device.device, "vkGetBufferDeviceAddress"));
  if (impl_->procs.get_buffer_device_address == nullptr) {
    impl_->procs.get_buffer_device_address = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
      vkGetDeviceProcAddr(impl_->device.device, "vkGetBufferDeviceAddressKHR"));
  }

  if (impl_->synchronization != detail::synchronization_backend::legacy) {
    char const *name = impl_->synchronization == detail::synchronization_backend::synchronization2_core
                         ? "vkCmdPipelineBarrier2"
                         : "vkCmdPipelineBarrier2KHR";
    impl_->procs.cmd_pipeline_barrier2 =
      reinterpret_cast<PFN_vkCmdPipelineBarrier2>(vkGetDeviceProcAddr(impl_->device.device, name));
    if (impl_->procs.cmd_pipeline_barrier2 == nullptr) {
      return fail(errc::unsupported, "synchronization2 command is unavailable");
    }
    char const *submit_name = impl_->synchronization == detail::synchronization_backend::synchronization2_core
                                ? "vkQueueSubmit2"
                                : "vkQueueSubmit2KHR";
    impl_->procs.queue_submit2 =
      reinterpret_cast<PFN_vkQueueSubmit2>(vkGetDeviceProcAddr(impl_->device.device, submit_name));
    if (impl_->procs.queue_submit2 == nullptr) {
      return fail(errc::unsupported, "synchronization2 queue submission is unavailable");
    }
  }
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
  return {};
}

}// namespace vkexec
