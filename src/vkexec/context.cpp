#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/device_procs.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include "detail/completion_waiter.hpp"
#include "detail/host_agent.hpp"

#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace vkexec {

struct context::impl
{
  vulkan_requirements requirements{};
  std::uint32_t api_version{ VK_API_VERSION_1_0 };
  device_procs procs{};
  vkb::Instance instance{};
  vkb::PhysicalDevice physical_device{};
  vkb::Device device{};
  VmaAllocator allocator{ VK_NULL_HANDLE };
  VkQueue compute_queue{ VK_NULL_HANDLE };
  VkQueue graphics_queue{ VK_NULL_HANDLE };
  VkQueue present_queue{ VK_NULL_HANDLE };
  std::uint32_t queue_family{ 0 };
  std::uint32_t graphics_family{ 0 };
  std::uint32_t present_family{ 0 };
  VkCommandPool command_pool{ VK_NULL_HANDLE };
  std::unique_ptr<detail::completion_waiter> completion_waiter;
  std::unique_ptr<detail::host_agent> host_agent;
  mutable std::mutex host_mutex;
  bool presentation_enabled{ false };
  bool has_instance{ false };
  bool has_device{ false };
  bool owns_instance{ false };
  bool owns_device{ false };
  bool owns_allocator{ false };
};

namespace {

  // Raise the request to the library floor so selectors never ask for less than vkexec needs.
  auto resolve_api_version(vulkan_requirements const &requirements) -> std::uint32_t
  {
    auto const requested = VK_MAKE_API_VERSION(0, requirements.api_version_major, requirements.api_version_minor, 0);
    auto const minimum =
      VK_MAKE_API_VERSION(0, vulkan_library::k_min_api_version_major, vulkan_library::k_min_api_version_minor, 0);
    return requested < minimum ? minimum : requested;
  }

  auto append_unique(std::vector<char const *> &dst, std::span<char const *const> src) -> void
  {
    // Pointer identity is enough: callers pass string literals / stable extension name macros.
    for (char const *extension : src) {
      if (extension == nullptr) { continue; }
      bool const exists = std::ranges::any_of(
        dst, [extension](char const *item) -> bool { return item != nullptr && std::strcmp(item, extension) == 0; });
      if (!exists) { dst.push_back(extension); }
    }
  }

  auto merge_instance_extensions(vulkan_requirements const &requirements,
    std::span<char const *const> extra_instance_extensions) -> std::vector<char const *>
  {
    // Library baselines first, then user extras; never remove required floors.
    std::vector<char const *> merged;
    append_unique(merged, vulkan_library::required_instance_extensions());
    append_unique(merged, requirements.instance_extensions);
    append_unique(merged, extra_instance_extensions);
    return merged;
  }

  auto merge_device_extensions(vulkan_requirements const &requirements, bool want_present) -> std::vector<char const *>
  {
    std::vector<char const *> merged;
    append_unique(merged, vulkan_library::required_device_extensions());
    if (want_present) { append_unique(merged, vulkan_library::required_presentation_device_extensions()); }
    append_unique(merged, requirements.device_extensions);
    return merged;
  }

  auto merge_features(vulkan_requirements const &requirements) -> VkPhysicalDeviceFeatures
  {
    (void)vulkan_library::required_features();
    return requirements.features;
  }

  auto configure_instance_builder(vkb::InstanceBuilder &builder,
    scheduler_options const &opts,
    std::uint32_t api_version,
    std::span<char const *const> extra_instance_extensions) -> void
  {
    builder.set_app_name("vkexec").set_engine_name("vkexec").require_api_version(
      VK_API_VERSION_MAJOR(api_version), VK_API_VERSION_MINOR(api_version));
    if (opts.validation_layers) { builder.enable_validation_layers().use_default_debug_messenger(); }

    auto const instance_exts = merge_instance_extensions(opts.requirements, extra_instance_extensions);
    if (!instance_exts.empty()) { builder.enable_extensions(instance_exts.size(), instance_exts.data()); }
  }

  auto configure_device_selector(vkb::PhysicalDeviceSelector &selector,
    vulkan_requirements const &requirements,
    std::uint32_t api_version,
    bool want_present) -> void
  {
    selector.set_minimum_version(VK_API_VERSION_MAJOR(api_version), VK_API_VERSION_MINOR(api_version))
      .require_present(want_present);

    auto const device_exts = merge_device_extensions(requirements, want_present);
    if (!device_exts.empty()) { selector.add_required_extensions(device_exts.size(), device_exts.data()); }

    selector.set_required_features(merge_features(requirements));

    for (extension_feature const &feature : requirements.required_extension_features) { feature.require(selector); }
  }

  auto apply_optional_device_requests(vkb::PhysicalDevice &physical_device, vulkan_requirements const &requirements)
    -> void
  {
    for (char const *extension : requirements.optional_device_extensions) {
      if (extension != nullptr) { (void)physical_device.enable_extension_if_present(extension); }
    }
    for (extension_feature const &feature : requirements.optional_extension_features) {
      (void)feature.enable_if_present(physical_device);
    }
  }

  auto build_headless_instance(scheduler_options const &opts, std::uint32_t api_version) -> result<vkb::Instance>
  {
    vkb::InstanceBuilder builder{};
    configure_instance_builder(builder, opts, api_version, {});
    builder.set_headless();
    auto const built = builder.build();
    if (!built) { return fail(make_error_from_vkb(built, "vk-bootstrap InstanceBuilder")); }
    return vkb_take(built);
  }

  auto build_instance_with_extensions(scheduler_options const &opts,
    std::uint32_t api_version,
    std::vector<char const *> const &extra_instance_extensions) -> result<vkb::Instance>
  {
    vkb::InstanceBuilder builder{};
    configure_instance_builder(builder, opts, api_version, extra_instance_extensions);
    builder.set_headless();
    auto const built = builder.build();
    if (!built) { return fail(make_error_from_vkb(built, "vk-bootstrap InstanceBuilder")); }
    return vkb_take(built);
  }

  auto select_physical_device(vkb::Instance const &instance,
    vulkan_requirements const &requirements,
    std::uint32_t api_version,
    VkSurfaceKHR surface,
    bool want_present) -> result<vkb::PhysicalDevice>
  {
    vkb::PhysicalDeviceSelector selector{ instance };
    configure_device_selector(selector, requirements, api_version, want_present);
    if (surface != VK_NULL_HANDLE) { selector.set_surface(surface); }
    auto const selected = selector.select();
    if (!selected) { return fail(make_error_from_vkb(selected, "vk-bootstrap PhysicalDeviceSelector")); }
    vkb::PhysicalDevice physical_device = vkb_take(selected);
    apply_optional_device_requests(physical_device, requirements);
    return physical_device;
  }

  auto build_device_into(vkb::PhysicalDevice const &physical_device, vkb::Device &out) -> status
  {
    auto const built = vkb::DeviceBuilder{ physical_device }.build();
    if (!built) { return fail(make_error_from_vkb(built, "vk-bootstrap DeviceBuilder")); }
    out = vkb_take(built);
    return {};
  }

}// namespace

context::context(factory_access /*access*/, [[maybe_unused]] uninitialized_tag tag) noexcept
  : impl_(std::make_unique<impl>())
{}

auto factory::make_context_t::operator()(scheduler_options const &opts) const
  -> sender<std::unique_ptr<::vkexec::context>>
{
  // Factory may allocate; sender::start() catches and maps to set_error.
  // NOLINTNEXTLINE(bugprone-exception-escape)
  return make_sender<std::unique_ptr<::vkexec::context>>([opts]() -> result<std::unique_ptr<::vkexec::context>> {
    auto ctx =
      std::make_unique<::vkexec::context>(::vkexec::context::factory_access{}, ::vkexec::context::uninitialized_tag{});

    VKEXEC_TRY(ctx->init_headless(opts));
    return ctx;
  });
}

auto factory::adopt_context_t::operator()(context_adopt_info const &info) const
  -> sender<std::unique_ptr<::vkexec::context>>
{
  // Factory may allocate; sender::start() catches and maps to set_error.
  // NOLINTNEXTLINE(bugprone-exception-escape)
  return make_sender<std::unique_ptr<::vkexec::context>>([info]() -> result<std::unique_ptr<::vkexec::context>> {
    auto ctx =
      std::make_unique<::vkexec::context>(::vkexec::context::factory_access{}, ::vkexec::context::uninitialized_tag{});

    VKEXEC_TRY(ctx->init_adopted(info));
    return ctx;
  });
}

auto context::instance() const noexcept -> VkInstance { return impl_->instance.instance; }
auto context::physical_device() const noexcept -> VkPhysicalDevice { return impl_->physical_device.physical_device; }
auto context::device() const noexcept -> VkDevice { return impl_->device.device; }
auto context::vkb_device() noexcept -> vkb::Device & { return impl_->device; }
auto context::vkb_device() const noexcept -> vkb::Device const & { return impl_->device; }
auto context::compute_queue() const noexcept -> VkQueue { return impl_->compute_queue; }
auto context::graphics_queue() const noexcept -> VkQueue { return impl_->graphics_queue; }
auto context::present_queue() const noexcept -> VkQueue { return impl_->present_queue; }
auto context::queue_family() const noexcept -> std::uint32_t { return impl_->queue_family; }
auto context::graphics_queue_family() const noexcept -> std::uint32_t { return impl_->graphics_family; }
auto context::present_queue_family() const noexcept -> std::uint32_t { return impl_->present_family; }
auto context::command_pool() const noexcept -> VkCommandPool { return impl_->command_pool; }
auto context::allocator() const noexcept -> VmaAllocator { return impl_->allocator; }
auto context::owns_instance() const noexcept -> bool { return impl_->owns_instance; }
auto context::owns_device() const noexcept -> bool { return impl_->owns_device; }
auto context::owns_allocator() const noexcept -> bool { return impl_->owns_allocator; }
auto context::presentation_enabled() const noexcept -> bool { return impl_->presentation_enabled; }
auto context::requirements() const noexcept -> vulkan_requirements const & { return impl_->requirements; }
auto context::api_version() const noexcept -> std::uint32_t { return impl_->api_version; }
auto context::procs() const noexcept -> device_procs const & { return impl_->procs; }

auto context::init_common_resources() -> status
{
  // Shared path for create() and window surface completion after the device exists.
  load_device_procs();
  VKEXEC_TRY(create_command_pool());
  VKEXEC_TRY(create_allocator());
  impl_->completion_waiter = std::make_unique<detail::completion_waiter>(impl_->device.device, impl_->compute_queue);
  impl_->host_agent = std::make_unique<detail::host_agent>();
  return {};
}

auto context::init_headless(scheduler_options const &opts) -> status
{
  impl_->requirements = opts.requirements;
  impl_->api_version = resolve_api_version(impl_->requirements);

  VKEXEC_TRY_ASSIGN(built_instance, build_headless_instance(opts, impl_->api_version));
  impl_->instance = built_instance;
  impl_->has_instance = true;
  impl_->owns_instance = true;

  VKEXEC_TRY_ASSIGN(selected_physical,
    select_physical_device(impl_->instance, impl_->requirements, impl_->api_version, VK_NULL_HANDLE, false));
  impl_->physical_device = std::move(selected_physical);

  VKEXEC_TRY(build_device_into(impl_->physical_device, impl_->device));
  impl_->has_device = true;
  impl_->owns_device = true;
  impl_->owns_allocator = true;

  VKEXEC_TRY(fetch_queues(false));
  return init_common_resources();
}

auto context::init_adopted(context_adopt_info const &info) -> status
{
  // Borrowed handles: never set owns_* for instance/device; allocator only if we create it.
  if (info.device == VK_NULL_HANDLE) { return fail(errc::invalid_argument, "context::adopt requires a VkDevice"); }
  if (info.compute_queue == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "context::adopt requires a compute VkQueue");
  }

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

  load_device_procs();

  if (info.allocator != VK_NULL_HANDLE) {
    impl_->allocator = info.allocator;
  } else {
    if (info.instance == VK_NULL_HANDLE || info.physical_device == VK_NULL_HANDLE) {
      return fail(
        errc::invalid_argument, "context::adopt requires instance and physical_device when allocator is null");
    }
    VKEXEC_TRY(create_allocator());
    impl_->owns_allocator = true;
  }

  VKEXEC_TRY(create_command_pool());
  impl_->completion_waiter = std::make_unique<detail::completion_waiter>(impl_->device.device, impl_->compute_queue);
  impl_->host_agent = std::make_unique<detail::host_agent>();
  return {};
}

context::context(factory_access /*access*/,
  instance_only_tag tag,
  scheduler_options const &opts,
  std::vector<char const *> const &instance_extensions)
  : impl_(std::make_unique<impl>())
{
  (void)tag;
  impl_->requirements = opts.requirements;
  impl_->api_version = resolve_api_version(impl_->requirements);
  auto built_instance = build_instance_with_extensions(opts, impl_->api_version, instance_extensions);
  if (!built_instance) { detail::contract_violation("context instance-only construction failed"); }
  impl_->instance = expected_take(built_instance);
  impl_->has_instance = true;
  impl_->owns_instance = true;
}

auto context::complete_for_surface(VkSurfaceKHR surface) -> status
{
  // Second-phase init used by window: instance already exists; select a present-capable device.
  if (surface == VK_NULL_HANDLE) { return fail(errc::invalid_argument, "complete_for_surface requires a surface"); }

  VKEXEC_TRY_ASSIGN(
    selected_physical, select_physical_device(impl_->instance, impl_->requirements, impl_->api_version, surface, true));
  impl_->physical_device = std::move(selected_physical);

  VKEXEC_TRY(build_device_into(impl_->physical_device, impl_->device));
  impl_->has_device = true;
  impl_->owns_device = true;
  impl_->owns_allocator = true;

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

auto context::load_device_procs() -> void
{
  impl_->procs = {};
  if (impl_->device.device == VK_NULL_HANDLE) { return; }

  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  impl_->procs.get_buffer_device_address = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
    vkGetDeviceProcAddr(impl_->device.device, "vkGetBufferDeviceAddress"));
  if (impl_->procs.get_buffer_device_address == nullptr) {
    impl_->procs.get_buffer_device_address = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
      vkGetDeviceProcAddr(impl_->device.device, "vkGetBufferDeviceAddressKHR"));
  }

  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
}

context::~context()
{
  // Agents first so outstanding completions finish before tearing down Vulkan objects.
  impl_->host_agent.reset();
  impl_->completion_waiter.reset();

  if (impl_->device.device != VK_NULL_HANDLE) {
    if (impl_->owns_device) { vkDeviceWaitIdle(impl_->device.device); }
    if (impl_->command_pool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(impl_->device.device, impl_->command_pool, nullptr);
      impl_->command_pool = VK_NULL_HANDLE;
    }
    if (impl_->owns_allocator && impl_->allocator != VK_NULL_HANDLE) {
      vmaDestroyAllocator(impl_->allocator);
      impl_->allocator = VK_NULL_HANDLE;
    }
    if (impl_->owns_device) {
      vkb::destroy_device(impl_->device);
      impl_->has_device = false;
    }
  }

  if (impl_->owns_instance && impl_->has_instance) {
    vkb::destroy_instance(impl_->instance);
    impl_->has_instance = false;
  }
}

auto context::create_command_pool() -> status
{
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = impl_->queue_family;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  if (vkCreateCommandPool(impl_->device.device, &pool_info, nullptr, &impl_->command_pool) != VK_SUCCESS) {
    return fail(VK_ERROR_UNKNOWN, "vkCreateCommandPool failed");
  }
  return {};
}

auto context::create_allocator() -> status
{
  VmaAllocatorCreateInfo allocator_info{};
  allocator_info.physicalDevice = impl_->physical_device.physical_device;
  allocator_info.device = impl_->device.device;
  allocator_info.instance = impl_->instance.instance;
  allocator_info.vulkanApiVersion = impl_->api_version;
  if (impl_->procs.get_buffer_device_address != nullptr) {
    allocator_info.flags |=
      static_cast<decltype(allocator_info.flags)>(VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT);
  }
  if (vmaCreateAllocator(&allocator_info, &impl_->allocator) != VK_SUCCESS) {
    return fail(VK_ERROR_UNKNOWN, "vmaCreateAllocator failed");
  }
  return {};
}

auto context::lock_host() const -> std::unique_lock<std::mutex> { return std::unique_lock{ impl_->host_mutex }; }

auto context::host_agent_thread_id() -> std::thread::id
{
  if (!impl_->host_agent) { impl_->host_agent = std::make_unique<detail::host_agent>(); }
  return impl_->host_agent->thread_id();
}

auto context::do_enqueue_fence_wait(VkSemaphore semaphore,
  VkFence fence,
  std::function<bool()> stop_requested,
  std::function<void(std::optional<error>, bool)> on_done) -> status
{
  if (!impl_->completion_waiter && impl_->device.device != VK_NULL_HANDLE) {
    impl_->completion_waiter = std::make_unique<detail::completion_waiter>(impl_->device.device, impl_->compute_queue);
  }
  if (!impl_->completion_waiter) {
    error failure = make_error(errc::invalid_argument, "completion waiter requires a VkDevice");
    status result = fail(failure);
    // From here on, ownership of the sync objects is consumed and no error/status
    // construction is allowed to occur before completion is delivered.
    if (fence != VK_NULL_HANDLE) {
      (void)vkWaitForFences(device(), 1, &fence, VK_TRUE, UINT64_MAX);
    } else if (compute_queue() != VK_NULL_HANDLE) {
      (void)vkQueueWaitIdle(compute_queue());
    }
    if (semaphore != VK_NULL_HANDLE) { vkDestroySemaphore(device(), semaphore, nullptr); }
    if (fence != VK_NULL_HANDLE) { vkDestroyFence(device(), fence, nullptr); }
    if (on_done) { on_done(std::move(failure), false); }
    return result;
  }
  detail::completion_waiter::stop_fn stop;
  if (stop_requested) { stop = std::move(stop_requested); }
  detail::completion_waiter::done_fn done;
  if (on_done) { done = std::move(on_done); }
  return impl_->completion_waiter->enqueue(semaphore, fence, std::move(stop), std::move(done));
}

auto context::do_enqueue_borrowed_fence_wait(VkFence fence,
  std::function<bool()> stop_requested,
  std::function<void(std::optional<error>, bool)> on_done) -> status
{
  if (!impl_->completion_waiter && impl_->device.device != VK_NULL_HANDLE) {
    impl_->completion_waiter = std::make_unique<detail::completion_waiter>(impl_->device.device, impl_->compute_queue);
  }
  if (!impl_->completion_waiter) {
    error const failure = make_error(errc::invalid_argument, "completion waiter requires a VkDevice");
    if (on_done) { on_done(failure, false); }
    return fail(failure);
  }
  detail::completion_waiter::stop_fn stop;
  if (stop_requested) { stop = std::move(stop_requested); }
  detail::completion_waiter::done_fn done;
  if (on_done) { done = std::move(on_done); }
  return impl_->completion_waiter->enqueue_borrowed(fence, std::move(stop), std::move(done));
}

auto context::do_enqueue_host(std::function<void()> task) -> status
{
  if (!impl_->host_agent) { impl_->host_agent = std::make_unique<detail::host_agent>(); }
  return impl_->host_agent->enqueue(std::move(task));
}

auto context::allocate_command_buffer() -> result<VkCommandBuffer>
{
  std::scoped_lock const lock(impl_->host_mutex);
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = impl_->command_pool;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  VkCommandBuffer cmd{ VK_NULL_HANDLE };
  if (VkResult const result = vkAllocateCommandBuffers(impl_->device.device, &alloc_info, &cmd); result != VK_SUCCESS) {
    return fail(result, "vkAllocateCommandBuffers failed");
  }
  return cmd;
}

auto context::free_command_buffer(VkCommandBuffer cmd) -> void
{
  std::scoped_lock const lock(impl_->host_mutex);
  vkFreeCommandBuffers(impl_->device.device, impl_->command_pool, 1, &cmd);
}

auto context::submit_and_wait(VkCommandBuffer cmd) -> status
{
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence{ VK_NULL_HANDLE };
  if (VkResult const create_result = vkCreateFence(impl_->device.device, &fence_info, nullptr, &fence);
    create_result != VK_SUCCESS) {
    return fail(create_result, "vkCreateFence failed");
  }

  {
    std::scoped_lock const lock(impl_->host_mutex);
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    if (VkResult const submit_result = vkQueueSubmit(impl_->compute_queue, 1, &submit_info, fence);
      submit_result != VK_SUCCESS) {
      vkDestroyFence(impl_->device.device, fence, nullptr);
      return fail(submit_result, "vkQueueSubmit failed");
    }
  }
  if (VkResult const wait_result = vkWaitForFences(impl_->device.device, 1, &fence, VK_TRUE, UINT64_MAX);
    wait_result != VK_SUCCESS) {
    vkDestroyFence(impl_->device.device, fence, nullptr);
    return fail(wait_result, "vkWaitForFences failed");
  }
  vkDestroyFence(impl_->device.device, fence, nullptr);
  return {};
}

auto context::submit_async(VkCommandBuffer cmd, VkSemaphore *out_semaphore, VkFence *out_fence) -> status
{
  if (out_semaphore == nullptr) { return fail(errc::invalid_argument, "submit_async requires out_semaphore"); }

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkSemaphore sem{ VK_NULL_HANDLE };
  if (VkResult const sem_result = vkCreateSemaphore(impl_->device.device, &semaphore_info, nullptr, &sem);
    sem_result != VK_SUCCESS) {
    return fail(sem_result, "vkCreateSemaphore failed");
  }

  VkFence fence{ VK_NULL_HANDLE };
  if (out_fence != nullptr) {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (VkResult const fence_result = vkCreateFence(impl_->device.device, &fence_info, nullptr, &fence);
      fence_result != VK_SUCCESS) {
      vkDestroySemaphore(impl_->device.device, sem, nullptr);
      return fail(fence_result, "vkCreateFence failed");
    }
    *out_fence = fence;
  }

  {
    std::scoped_lock const lock(impl_->host_mutex);
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &sem;
    if (VkResult const submit_result = vkQueueSubmit(impl_->compute_queue, 1, &submit_info, fence);
      submit_result != VK_SUCCESS) {
      vkDestroySemaphore(impl_->device.device, sem, nullptr);
      if (fence != VK_NULL_HANDLE) { vkDestroyFence(impl_->device.device, fence, nullptr); }
      return fail(submit_result, "vkQueueSubmit failed");
    }
  }
  *out_semaphore = sem;
  return {};
}

auto context::submit(queue_submit const &info) const -> status
{
  if (info.command_buffers.empty()) {
    return fail(errc::invalid_argument, "queue_submit requires at least one command buffer");
  }

  VkQueue queue = info.queue != VK_NULL_HANDLE ? info.queue : impl_->compute_queue;
  if (queue == VK_NULL_HANDLE) { return fail(errc::invalid_argument, "queue_submit requires a VkQueue"); }

  std::vector<VkSemaphore> wait_semaphores;
  std::vector<VkPipelineStageFlags> wait_stages;
  std::vector<std::uint64_t> wait_values;
  wait_semaphores.reserve(info.waits.size());
  wait_stages.reserve(info.waits.size());
  wait_values.reserve(info.waits.size());
  for (semaphore_submit const &wait : info.waits) {
    if (wait.semaphore == VK_NULL_HANDLE) {
      return fail(errc::invalid_argument, "queue_submit wait semaphore is null");
    }
    wait_semaphores.push_back(wait.semaphore);
    wait_stages.push_back(wait.stage);
    wait_values.push_back(wait.value);
  }

  std::vector<VkSemaphore> signal_semaphores;
  std::vector<std::uint64_t> signal_values;
  signal_semaphores.reserve(info.signals.size());
  signal_values.reserve(info.signals.size());
  for (semaphore_submit const &signal : info.signals) {
    if (signal.semaphore == VK_NULL_HANDLE) {
      return fail(errc::invalid_argument, "queue_submit signal semaphore is null");
    }
    signal_semaphores.push_back(signal.semaphore);
    signal_values.push_back(signal.value);
  }

  bool const use_timeline =
    std::ranges::any_of(info.waits, [](semaphore_submit const &entry) -> bool { return entry.value != 0; })
    || std::ranges::any_of(info.signals, [](semaphore_submit const &entry) -> bool { return entry.value != 0; });

  // Any non-zero value implies timeline; binary-only submits omit the pNext chain.
  VkTimelineSemaphoreSubmitInfo timeline_info{};
  timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  if (use_timeline) {
    timeline_info.waitSemaphoreValueCount = static_cast<std::uint32_t>(wait_values.size());
    timeline_info.pWaitSemaphoreValues = wait_values.empty() ? nullptr : wait_values.data();
    timeline_info.signalSemaphoreValueCount = static_cast<std::uint32_t>(signal_values.size());
    timeline_info.pSignalSemaphoreValues = signal_values.empty() ? nullptr : signal_values.data();
  }

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.pNext = use_timeline ? &timeline_info : nullptr;
  submit_info.waitSemaphoreCount = static_cast<std::uint32_t>(wait_semaphores.size());
  submit_info.pWaitSemaphores = wait_semaphores.empty() ? nullptr : wait_semaphores.data();
  submit_info.pWaitDstStageMask = wait_stages.empty() ? nullptr : wait_stages.data();
  submit_info.commandBufferCount = static_cast<std::uint32_t>(info.command_buffers.size());
  submit_info.pCommandBuffers = info.command_buffers.data();
  submit_info.signalSemaphoreCount = static_cast<std::uint32_t>(signal_semaphores.size());
  submit_info.pSignalSemaphores = signal_semaphores.empty() ? nullptr : signal_semaphores.data();

  std::scoped_lock const lock(impl_->host_mutex);
  if (VkResult const submit_result = vkQueueSubmit(queue, 1, &submit_info, info.fence); submit_result != VK_SUCCESS) {
    return fail(submit_result, "vkQueueSubmit failed");
  }
  return {};
}

}// namespace vkexec
