#include <vkexec/detail/move_only_function.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/result.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include "completion_waiter.hpp"
#include "host_agent.hpp"
#include "pipeline_cache.hpp"

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

namespace {

  auto resolve_api_version(vulkan_requirements const &requirements) -> std::uint32_t
  {
    auto const requested = VK_MAKE_API_VERSION(0, requirements.api_version_major, requirements.api_version_minor, 0);
    auto const minimum =
      VK_MAKE_API_VERSION(0, vulkan_library::k_min_api_version_major, vulkan_library::k_min_api_version_minor, 0);
    return requested < minimum ? minimum : requested;
  }

  auto append_unique(std::vector<char const *> &dst, std::span<char const *const> src) -> void
  {
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

context::context([[maybe_unused]] uninitialized_tag tag) noexcept {}

auto context::create(scheduler_options const &opts) -> detail::sync_sender_fn<std::unique_ptr<context>>
{
  return detail::make_sync_sender_fn<std::unique_ptr<context>>([opts]() -> result<std::unique_ptr<context>> {
    auto ctx = std::unique_ptr<context>(new context(uninitialized_tag{}));
    VKEXEC_TRY(ctx->init_headless(opts));
    return ctx;
  });
}

auto context::adopt(context_adopt_info const &info) -> detail::sync_sender_fn<std::unique_ptr<context>>
{
  return detail::make_sync_sender_fn<std::unique_ptr<context>>([info]() -> result<std::unique_ptr<context>> {
    auto ctx = std::unique_ptr<context>(new context(uninitialized_tag{}));
    VKEXEC_TRY(ctx->init_adopted(info));
    return ctx;
  });
}

auto context::init_common_resources() -> status
{
  load_device_procs();
  VKEXEC_TRY(create_command_pool());
  VKEXEC_TRY(create_allocator());
  pipeline_cache_ = std::make_unique<pipeline_cache>(*this);
  completion_waiter_ = std::make_unique<detail::completion_waiter>(device_.device, compute_queue_);
  host_agent_ = std::make_unique<detail::host_agent>();
  return {};
}

auto context::init_headless(scheduler_options const &opts) -> status
{
  requirements_ = opts.requirements;
  api_version_ = resolve_api_version(requirements_);

  VKEXEC_TRY_ASSIGN(built_instance, build_headless_instance(opts, api_version_));
  instance_ = built_instance;
  has_instance_ = true;
  owns_instance_ = true;

  VKEXEC_TRY_ASSIGN(
    selected_physical, select_physical_device(instance_, requirements_, api_version_, VK_NULL_HANDLE, false));
  physical_device_ = std::move(selected_physical);

  VKEXEC_TRY(build_device_into(physical_device_, device_));
  has_device_ = true;
  owns_device_ = true;
  owns_allocator_ = true;

  VKEXEC_TRY(fetch_queues(false));
  return init_common_resources();
}

auto context::init_adopted(context_adopt_info const &info) -> status
{
  if (info.device == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "context::adopt requires a VkDevice");
  }
  if (info.compute_queue == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "context::adopt requires a compute VkQueue");
  }

  instance_.instance = info.instance;
  physical_device_.physical_device = info.physical_device;
  device_.device = info.device;
  has_instance_ = info.instance != VK_NULL_HANDLE;
  has_device_ = true;

  compute_queue_ = info.compute_queue;
  queue_family_ = info.compute_queue_family;

  if (info.graphics_queue != VK_NULL_HANDLE) {
    graphics_queue_ = info.graphics_queue;
    graphics_family_ = info.graphics_queue_family;
  } else {
    graphics_queue_ = compute_queue_;
    graphics_family_ = queue_family_;
  }

  if (info.present_queue != VK_NULL_HANDLE) {
    present_queue_ = info.present_queue;
    present_family_ = info.present_queue_family;
  } else {
    present_queue_ = graphics_queue_;
    present_family_ = graphics_family_;
  }

  load_device_procs();

  if (info.allocator != VK_NULL_HANDLE) {
    allocator_ = info.allocator;
  } else {
    if (info.instance == VK_NULL_HANDLE || info.physical_device == VK_NULL_HANDLE) {
      return fail(
        errc::invalid_argument, "context::adopt requires instance and physical_device when allocator is null");
    }
    VKEXEC_TRY(create_allocator());
    owns_allocator_ = true;
  }

  VKEXEC_TRY(create_command_pool());
  pipeline_cache_ = std::make_unique<pipeline_cache>(*this);
  completion_waiter_ = std::make_unique<detail::completion_waiter>(device_.device, compute_queue_);
  host_agent_ = std::make_unique<detail::host_agent>();
  return {};
}

context::context(instance_only_tag tag,
  scheduler_options const &opts,
  std::vector<char const *> const &instance_extensions)
  : requirements_(opts.requirements), api_version_(resolve_api_version(requirements_))
{
  (void)tag;
  auto built_instance = build_instance_with_extensions(opts, api_version_, instance_extensions);
  if (!built_instance) { detail::contract_violation("context instance-only construction failed"); }
  instance_ = expected_take(built_instance);
  has_instance_ = true;
  owns_instance_ = true;
}

auto context::complete_for_surface(VkSurfaceKHR surface) -> status
{
  if (surface == VK_NULL_HANDLE) {
    return fail(errc::invalid_argument, "complete_for_surface requires a surface");
  }

  VKEXEC_TRY_ASSIGN(selected_physical, select_physical_device(instance_, requirements_, api_version_, surface, true));
  physical_device_ = std::move(selected_physical);

  VKEXEC_TRY(build_device_into(physical_device_, device_));
  has_device_ = true;
  owns_device_ = true;
  owns_allocator_ = true;

  VKEXEC_TRY(fetch_queues(true));
  VKEXEC_TRY(init_common_resources());
  presentation_enabled_ = true;
  return {};
}

auto context::fetch_queues(bool want_present) -> status
{
  if (auto graphics = device_.get_queue_and_index(vkb::QueueType::graphics)) {
    graphics_queue_ = graphics->first;
    graphics_family_ = graphics->second;
  }

  if (auto compute = device_.get_queue_and_index(vkb::QueueType::compute)) {
    compute_queue_ = compute->first;
    queue_family_ = compute->second;
  } else if (graphics_queue_ != VK_NULL_HANDLE) {
    compute_queue_ = graphics_queue_;
    queue_family_ = graphics_family_;
  } else {
    return fail(errc::unsupported, "no compute or graphics queue available");
  }

  if (graphics_queue_ == VK_NULL_HANDLE) {
    graphics_queue_ = compute_queue_;
    graphics_family_ = queue_family_;
  }

  if (want_present) {
    auto present = device_.get_queue_and_index(vkb::QueueType::present);
    if (!present) { return fail(errc::unsupported, "no present queue available"); }
    present_queue_ = present->first;
    present_family_ = present->second;
  } else {
    present_queue_ = graphics_queue_;
    present_family_ = graphics_family_;
  }
  return {};
}

auto context::load_device_procs() -> void
{
  procs_ = {};
  if (device_.device == VK_NULL_HANDLE) { return; }

  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  procs_.get_buffer_device_address =
    reinterpret_cast<PFN_vkGetBufferDeviceAddress>(vkGetDeviceProcAddr(device_.device, "vkGetBufferDeviceAddress"));
  if (procs_.get_buffer_device_address == nullptr) {
    procs_.get_buffer_device_address = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
      vkGetDeviceProcAddr(device_.device, "vkGetBufferDeviceAddressKHR"));
  }

  procs_.write_resource_descriptors = reinterpret_cast<PFN_vkWriteResourceDescriptorsEXT>(
    vkGetDeviceProcAddr(device_.device, "vkWriteResourceDescriptorsEXT"));
  procs_.write_sampler_descriptors = reinterpret_cast<PFN_vkWriteSamplerDescriptorsEXT>(
    vkGetDeviceProcAddr(device_.device, "vkWriteSamplerDescriptorsEXT"));
  procs_.cmd_bind_resource_heap =
    reinterpret_cast<PFN_vkCmdBindResourceHeapEXT>(vkGetDeviceProcAddr(device_.device, "vkCmdBindResourceHeapEXT"));
  procs_.cmd_bind_sampler_heap =
    reinterpret_cast<PFN_vkCmdBindSamplerHeapEXT>(vkGetDeviceProcAddr(device_.device, "vkCmdBindSamplerHeapEXT"));
  procs_.cmd_push_data =
    reinterpret_cast<PFN_vkCmdPushDataEXT>(vkGetDeviceProcAddr(device_.device, "vkCmdPushDataEXT"));
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
}

context::~context()
{
  host_agent_.reset();
  completion_waiter_.reset();
  pipeline_cache_.reset();

  if (device_.device != VK_NULL_HANDLE) {
    if (owns_device_) { vkDeviceWaitIdle(device_.device); }
    if (command_pool_ != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device_.device, command_pool_, nullptr);
      command_pool_ = VK_NULL_HANDLE;
    }
    if (owns_allocator_ && allocator_ != VK_NULL_HANDLE) {
      vmaDestroyAllocator(allocator_);
      allocator_ = VK_NULL_HANDLE;
    }
    if (owns_device_) {
      vkb::destroy_device(device_);
      has_device_ = false;
    }
  }

  if (owns_instance_ && has_instance_) {
    vkb::destroy_instance(instance_);
    has_instance_ = false;
  }
}

auto context::create_command_pool() -> status
{
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = queue_family_;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  if (vkCreateCommandPool(device_.device, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
    return fail(VK_ERROR_UNKNOWN, "vkCreateCommandPool failed");
  }
  return {};
}

auto context::create_allocator() -> status
{
  VmaAllocatorCreateInfo allocator_info{};
  allocator_info.physicalDevice = physical_device_.physical_device;
  allocator_info.device = device_.device;
  allocator_info.instance = instance_.instance;
  allocator_info.vulkanApiVersion = api_version_;
  if (procs_.get_buffer_device_address != nullptr) {
    allocator_info.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  }
  if (vmaCreateAllocator(&allocator_info, &allocator_) != VK_SUCCESS) {
    return fail(VK_ERROR_UNKNOWN, "vmaCreateAllocator failed");
  }
  return {};
}

auto context::lock_host() const -> std::unique_lock<std::mutex> { return std::unique_lock{ host_mutex_ }; }

auto context::ensure_completion_waiter() -> result<detail::completion_waiter *>
{
  if (!completion_waiter_) {
    if (device_.device == VK_NULL_HANDLE) {
      return fail(errc::invalid_argument, "completion waiter requires a VkDevice");
    }
    completion_waiter_ = std::make_unique<detail::completion_waiter>(device_.device, compute_queue_);
  }
  return completion_waiter_.get();
}

auto context::ensure_host_agent() -> result<detail::host_agent *>
{
  if (!host_agent_) { host_agent_ = std::make_unique<detail::host_agent>(); }
  return host_agent_.get();
}

auto context::host_agent_thread_id() -> std::thread::id
{
  auto agent = ensure_host_agent();
  if (!agent) { detail::contract_violation("host agent unavailable"); }
  return (*agent)->thread_id();
}

auto context::do_enqueue_fence_wait(VkSemaphore semaphore,
  VkFence fence,
  detail::move_only_function<bool()> stop_requested,
  detail::move_only_function<void(std::optional<error>, bool)> on_done) -> status
{
  auto waiter = ensure_completion_waiter();
  if (!waiter) {
    if (fence != VK_NULL_HANDLE) {
      (void)vkWaitForFences(device(), 1, &fence, VK_TRUE, UINT64_MAX);
    } else if (compute_queue() != VK_NULL_HANDLE) {
      (void)vkQueueWaitIdle(compute_queue());
    }
    if (semaphore != VK_NULL_HANDLE) { vkDestroySemaphore(device(), semaphore, nullptr); }
    if (fence != VK_NULL_HANDLE) { vkDestroyFence(device(), fence, nullptr); }
    if (on_done) { on_done(waiter.error(), false); }
    return fail(waiter);
  }
  return expected_take(waiter)->enqueue(semaphore, fence, std::move(stop_requested), std::move(on_done));
}

auto context::do_enqueue_borrowed_fence_wait(VkFence fence,
  detail::move_only_function<bool()> stop_requested,
  detail::move_only_function<void(std::optional<error>, bool)> on_done) -> status
{
  auto waiter = ensure_completion_waiter();
  if (!waiter) {
    if (on_done) { on_done(waiter.error(), false); }
    return fail(waiter);
  }
  return expected_take(waiter)->enqueue_borrowed(fence, std::move(stop_requested), std::move(on_done));
}

auto context::do_enqueue_host(detail::move_only_function<void()> task) -> status
{
  VKEXEC_TRY_ASSIGN(agent, ensure_host_agent());
  return agent->enqueue(std::move(task));
}

auto context::allocate_command_buffer() -> result<VkCommandBuffer>
{
  std::scoped_lock const lock(host_mutex_);
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = command_pool_;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  VkCommandBuffer cmd{ VK_NULL_HANDLE };
  if (VkResult const result = vkAllocateCommandBuffers(device_.device, &alloc_info, &cmd); result != VK_SUCCESS) {
    return fail(result, "vkAllocateCommandBuffers failed");
  }
  return cmd;
}

auto context::free_command_buffer(VkCommandBuffer cmd) -> void
{
  std::scoped_lock const lock(host_mutex_);
  vkFreeCommandBuffers(device_.device, command_pool_, 1, &cmd);
}

auto context::submit_and_wait(VkCommandBuffer cmd) -> status
{
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence{ VK_NULL_HANDLE };
  if (VkResult const create_result = vkCreateFence(device_.device, &fence_info, nullptr, &fence);
    create_result != VK_SUCCESS) {
    return fail(create_result, "vkCreateFence failed");
  }

  {
    std::scoped_lock const lock(host_mutex_);
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    if (VkResult const submit_result = vkQueueSubmit(compute_queue_, 1, &submit_info, fence);
      submit_result != VK_SUCCESS) {
      vkDestroyFence(device_.device, fence, nullptr);
      return fail(submit_result, "vkQueueSubmit failed");
    }
  }
  if (VkResult const wait_result = vkWaitForFences(device_.device, 1, &fence, VK_TRUE, UINT64_MAX);
    wait_result != VK_SUCCESS) {
    vkDestroyFence(device_.device, fence, nullptr);
    return fail(wait_result, "vkWaitForFences failed");
  }
  vkDestroyFence(device_.device, fence, nullptr);
  return {};
}

auto context::submit_async(VkCommandBuffer cmd, VkSemaphore *out_semaphore, VkFence *out_fence) -> status
{
  if (out_semaphore == nullptr) { return fail(errc::invalid_argument, "submit_async requires out_semaphore"); }

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VkSemaphore sem{ VK_NULL_HANDLE };
  if (VkResult const sem_result = vkCreateSemaphore(device_.device, &semaphore_info, nullptr, &sem);
    sem_result != VK_SUCCESS) {
    return fail(sem_result, "vkCreateSemaphore failed");
  }

  VkFence fence{ VK_NULL_HANDLE };
  if (out_fence != nullptr) {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (VkResult const fence_result = vkCreateFence(device_.device, &fence_info, nullptr, &fence);
      fence_result != VK_SUCCESS) {
      vkDestroySemaphore(device_.device, sem, nullptr);
      return fail(fence_result, "vkCreateFence failed");
    }
    *out_fence = fence;
  }

  {
    std::scoped_lock const lock(host_mutex_);
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &sem;
    if (VkResult const submit_result = vkQueueSubmit(compute_queue_, 1, &submit_info, fence);
      submit_result != VK_SUCCESS) {
      vkDestroySemaphore(device_.device, sem, nullptr);
      if (fence != VK_NULL_HANDLE) { vkDestroyFence(device_.device, fence, nullptr); }
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

  VkQueue queue = info.queue != VK_NULL_HANDLE ? info.queue : compute_queue_;
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

  std::scoped_lock const lock(host_mutex_);
  if (VkResult const submit_result = vkQueueSubmit(queue, 1, &submit_info, info.fence); submit_result != VK_SUCCESS) {
    return fail(submit_result, "vkQueueSubmit failed");
  }
  return {};
}

auto context::get_or_create_from_spirv(std::span<std::uint32_t const> spirv, layout_desc const &desc)
  -> result<std::reference_wrapper<pipeline_resources>>
{ return pipeline_cache_->get_or_create_from_spirv(spirv, desc); }

}// namespace vkexec
