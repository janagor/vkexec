#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/detail/capability_id.hpp>
#include <vkexec/detail/object_synchronization.hpp>
#include <vkexec/detail/submission.hpp>
#include <vkexec/detail/synchronization.hpp>
#include <vkexec/detail/vk_bootstrap_error.hpp>
#include <vkexec/detail/worker_callbacks.hpp>
#include <vkexec/device_capabilities.hpp>
#include <vkexec/device_procs.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include "detail/completion_waiter.hpp"
#include "detail/host_agent.hpp"
#include "detail/vk_bootstrap_feature.hpp"

#include <VkBootstrap.h>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vkexec {

struct detail::context_state
{
  vulkan_requirements creation_requirements{};
  device_capabilities capabilities{};
  device_procs procs{};
  detail::synchronization_backend synchronization{ detail::synchronization_backend::legacy };
  vkb::Instance instance{};
  vkb::PhysicalDevice physical_device{};
  vkb::Device device{};
  VkQueue compute_queue{ VK_NULL_HANDLE };
  VkQueue graphics_queue{ VK_NULL_HANDLE };
  VkQueue present_queue{ VK_NULL_HANDLE };
  std::uint32_t queue_family{ 0 };
  std::uint32_t graphics_family{ 0 };
  std::uint32_t present_family{ 0 };
  std::unique_ptr<detail::completion_waiter> completion_waiter;
  std::unique_ptr<detail::host_agent> host_agent;
  mutable detail::queue_synchronization_registry queue_synchronization;
  mutable detail::descriptor_pool_synchronization_registry descriptor_pool_synchronization;
  // A checked-out pool has exactly one host owner; only returned pools appear here.
  std::mutex command_pool_cache_mutex;
  std::vector<std::pair<std::uint32_t, VkCommandPool>> available_command_pools;
  std::mutex command_buffer_mutex;
  std::unordered_map<VkCommandBuffer, VkCommandPool> command_buffer_pools;
  bool presentation_enabled{ false };
  bool has_instance{ false };
  bool has_device{ false };
  bool owns_instance{ false };
  bool owns_device{ false };
  context_state *next_retired{ nullptr };
  std::shared_ptr<std::promise<void>> retirement_done{ std::make_shared<std::promise<void>>() };
  std::shared_future<void> retirement{ retirement_done->get_future().share() };

  ~context_state();
  context_state() = default;
  context_state(context_state const &) = delete;
  auto operator=(context_state const &) -> context_state & = delete;
  context_state(context_state &&) = delete;
  auto operator=(context_state &&) -> context_state & = delete;

  auto register_known_queues() -> status
  {
#if VKEXEC_HAS_EXCEPTIONS
    try {
#endif
      auto register_queue = [this](VkQueue queue) -> void {
        if (queue != VK_NULL_HANDLE) { (void)queue_synchronization.state(queue); }
      };
      register_queue(compute_queue);
      register_queue(graphics_queue);
      register_queue(present_queue);
#if VKEXEC_HAS_EXCEPTIONS
    } catch (...) {
      return fail(unexpected_exception_error());
    }
#endif
    return {};
  }
};

namespace {

  class retirement_executor
  {
  public:
    retirement_executor() : worker_([this]() noexcept -> void { run(); }) {}
    retirement_executor(retirement_executor const &) = delete;
    auto operator=(retirement_executor const &) -> retirement_executor & = delete;
    retirement_executor(retirement_executor &&) = delete;
    auto operator=(retirement_executor &&) -> retirement_executor & = delete;

    ~retirement_executor()
    {
      {
        std::scoped_lock const lock(mutex_);
        stopping_ = true;
      }
      cv_.notify_one();
      worker_.join();
    }

    auto retire(std::unique_ptr<detail::context_state> state) noexcept -> void
    {
      auto *retired = state.release();
      {
        std::scoped_lock const lock(mutex_);
        if (tail_ != nullptr) {
          tail_->next_retired = retired;
        } else {
          head_ = retired;
        }
        tail_ = retired;
      }
      cv_.notify_one();
    }

  private:
    auto run() noexcept -> void
    {
      for (;;) {
        std::unique_ptr<detail::context_state> state;
        {
          std::unique_lock lock(mutex_);
          cv_.wait(lock, [this]() noexcept -> bool { return stopping_ || head_ != nullptr; });
          if (head_ == nullptr && stopping_) { return; }
          state.reset(head_);
          head_ = state->next_retired;
          if (head_ == nullptr) { tail_ = nullptr; }
        }
        auto done = state->retirement_done;
        state.reset();
        done->set_value();
      }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    detail::context_state *head_{ nullptr };
    detail::context_state *tail_{ nullptr };
    bool stopping_{ false };
    std::thread worker_;
  };

  [[nodiscard]] auto runtime_reaper() -> retirement_executor &
  {
    static retirement_executor executor;
    return executor;
  }

  auto enable_capability(device_capabilities &caps, detail::capability_id capability) noexcept -> void
  {
    switch (capability) {
    case detail::capability_id::synchronization2:
      caps.synchronization2 = true;
      break;
    case detail::capability_id::timeline_semaphore:
      caps.timeline_semaphore = true;
      break;
    case detail::capability_id::buffer_device_address:
      caps.buffer_device_address = true;
      break;
    case detail::capability_id::dynamic_rendering:
      caps.dynamic_rendering = true;
      break;
    case detail::capability_id::none:
      break;
    }
  }

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

    for (extension_feature const &feature : requirements.required_extension_features) {
      detail::extension_feature_access::require(feature, selector);
    }
  }

  auto contains_extension(std::span<char const *const> extensions, char const *name) -> bool
  {
    return std::ranges::any_of(extensions,
      [name](char const *extension) -> bool { return extension != nullptr && std::strcmp(extension, name) == 0; });
  }

  auto instance_extension_available(char const *name) -> bool
  {
    std::uint32_t count = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS) { return false; }
    std::vector<VkExtensionProperties> extensions(count);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()) != VK_SUCCESS) { return false; }
    return std::ranges::any_of(extensions, [name](VkExtensionProperties const &extension) -> bool {
      return std::strcmp(std::data(extension.extensionName), name) == 0;
    });
  }

  auto configure_optional_synchronization2(vulkan_requirements &requirements, std::uint32_t api_version) -> void
  {
    auto const has_feature = [](std::vector<extension_feature> const &features) -> bool {
      return std::ranges::any_of(features, [](extension_feature const &feature) -> bool {
        return detail::extension_feature_access::s_type(feature)
               == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
      });
    };
    bool const feature_requested =
      has_feature(requirements.required_extension_features) || has_feature(requirements.optional_extension_features);
    if (api_version >= VK_API_VERSION_1_3) {
      if (!feature_requested) {
        VkPhysicalDeviceSynchronization2Features feature{};
        feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
        feature.synchronization2 = VK_TRUE;
        requirements.optional_extension_features.push_back(
          detail::extension_feature_access::make(feature, detail::capability_id::synchronization2));
      }
      return;
    }
    if (api_version < VK_API_VERSION_1_1
        && !contains_extension(
          requirements.instance_extensions, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
      if (!instance_extension_available(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) { return; }
      requirements.instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }
    if (!contains_extension(requirements.device_extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
        && !contains_extension(requirements.optional_device_extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)) {
      requirements.optional_device_extensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    }
    if (!feature_requested) {
      VkPhysicalDeviceSynchronization2FeaturesKHR feature{};
      feature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
      feature.synchronization2 = VK_TRUE;
      requirements.optional_extension_features.push_back(
        detail::extension_feature_access::make(feature, detail::capability_id::synchronization2));
    }
  }

  auto apply_optional_device_requests(vkb::PhysicalDevice &physical_device,
    vulkan_requirements const &requirements,
    device_capabilities &caps) -> void
  {
    bool khr_extension_enabled =
      contains_extension(requirements.device_extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    for (char const *extension : requirements.optional_device_extensions) {
      if (extension == nullptr) { continue; }
      if (contains_extension(requirements.device_extensions, extension)) { continue; }
      bool const enabled = physical_device.enable_extension_if_present(extension);
      if (std::strcmp(extension, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0) { khr_extension_enabled |= enabled; }
    }
    for (extension_feature const &feature : requirements.required_extension_features) {
      enable_capability(caps, detail::extension_feature_access::capability(feature));
    }
    for (extension_feature const &feature : requirements.optional_extension_features) {
      bool const already_required = std::ranges::any_of(
        requirements.required_extension_features, [&feature](extension_feature const &required) -> bool {
          auto const capability = detail::extension_feature_access::capability(feature);
          return capability != detail::capability_id::none
                   ? detail::extension_feature_access::capability(required) == capability
                   : detail::extension_feature_access::s_type(required)
                       == detail::extension_feature_access::s_type(feature);
        });
      if (already_required) { continue; }
      bool const synchronization2 =
        detail::extension_feature_access::capability(feature) == detail::capability_id::synchronization2;
      if (synchronization2 && caps.api_version < VK_API_VERSION_1_3 && !khr_extension_enabled) { continue; }
      bool const enabled = detail::extension_feature_access::enable_if_present(feature, physical_device);
      if (enabled) { enable_capability(caps, detail::extension_feature_access::capability(feature)); }
    }
  }

  auto build_headless_instance(scheduler_options const &opts, std::uint32_t api_version) -> result<vkb::Instance>
  {
    vkb::InstanceBuilder builder{};
    configure_instance_builder(builder, opts, api_version, {});
    builder.set_headless();
    auto const built = builder.build();
    if (!built) { return fail(detail::make_error_from_vkb(built, "vk-bootstrap InstanceBuilder")); }
    return detail::vkb_take(built);
  }

  auto build_instance_with_extensions(scheduler_options const &opts,
    std::uint32_t api_version,
    std::vector<char const *> const &extra_instance_extensions) -> result<vkb::Instance>
  {
    vkb::InstanceBuilder builder{};
    configure_instance_builder(builder, opts, api_version, extra_instance_extensions);
    builder.set_headless();
    auto const built = builder.build();
    if (!built) { return fail(detail::make_error_from_vkb(built, "vk-bootstrap InstanceBuilder")); }
    return detail::vkb_take(built);
  }

  auto select_physical_device(vkb::Instance const &instance,
    vulkan_requirements const &requirements,
    std::uint32_t api_version,
    VkSurfaceKHR surface,
    bool want_present,
    device_capabilities &caps) -> result<vkb::PhysicalDevice>
  {
    vkb::PhysicalDeviceSelector selector{ instance };
    configure_device_selector(selector, requirements, api_version, want_present);
    if (surface != VK_NULL_HANDLE) { selector.set_surface(surface); }
    auto const selected = selector.select();
    if (!selected) { return fail(detail::make_error_from_vkb(selected, "vk-bootstrap PhysicalDeviceSelector")); }
    vkb::PhysicalDevice physical_device = detail::vkb_take(selected);
    apply_optional_device_requests(physical_device, requirements, caps);
    return physical_device;
  }

  auto build_device_into(vkb::PhysicalDevice const &physical_device, vkb::Device &out) -> status
  {
    auto const built = vkb::DeviceBuilder{ physical_device }.build();
    if (!built) { return fail(detail::make_error_from_vkb(built, "vk-bootstrap DeviceBuilder")); }
    out = detail::vkb_take(built);
    return {};
  }

  [[nodiscard]] auto make_context_state() -> detail::context_handle
  {
    (void)runtime_reaper();
    return { new detail::context_state, [](detail::context_state *state) noexcept -> void {
      std::unique_ptr<detail::context_state> owned{ state };
      bool const on_worker = (owned->host_agent && owned->host_agent->on_agent_thread())
                             || (owned->completion_waiter && owned->completion_waiter->on_agent_thread());
      if (on_worker) {
        runtime_reaper().retire(std::move(owned));
      } else {
        auto done = owned->retirement_done;
        owned.reset();
        done->set_value();
      }
    } };
  }

}// namespace

context::context(factory_access /*access*/, [[maybe_unused]] uninitialized_tag tag)
  : impl_(make_context_state())
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
auto context::owns_instance() const noexcept -> bool { return impl_->owns_instance; }
auto context::owns_device() const noexcept -> bool { return impl_->owns_device; }
auto context::presentation_enabled() const noexcept -> bool { return impl_->presentation_enabled; }
auto context::capabilities() const noexcept -> device_capabilities const & { return impl_->capabilities; }
auto context::api_version() const noexcept -> std::uint32_t { return impl_->capabilities.api_version; }
auto context::procs() const noexcept -> device_procs const & { return impl_->procs; }
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
  impl_->capabilities.api_version = resolve_api_version(impl_->creation_requirements);
  configure_optional_synchronization2(impl_->creation_requirements, impl_->capabilities.api_version);

  scheduler_options effective_opts = opts;
  effective_opts.requirements = impl_->creation_requirements;
  VKEXEC_TRY_ASSIGN(built_instance, build_headless_instance(effective_opts, api_version()));
  impl_->instance = built_instance;
  impl_->has_instance = true;
  impl_->owns_instance = true;

  VKEXEC_TRY_ASSIGN(selected_physical,
    select_physical_device(
      impl_->instance, impl_->creation_requirements, api_version(), VK_NULL_HANDLE, false, impl_->capabilities));
  impl_->physical_device = std::move(selected_physical);
  impl_->synchronization = detail::select_synchronization_backend(impl_->capabilities);


  VKEXEC_TRY(build_device_into(impl_->physical_device, impl_->device));
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
  : impl_(make_context_state())
{
  (void)tag;
  impl_->creation_requirements = opts.requirements;
  impl_->capabilities.api_version = resolve_api_version(impl_->creation_requirements);
  configure_optional_synchronization2(impl_->creation_requirements, api_version());
  scheduler_options effective_opts = opts;
  effective_opts.requirements = impl_->creation_requirements;
  auto built_instance = build_instance_with_extensions(effective_opts, api_version(), instance_extensions);
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
    select_physical_device(
      impl_->instance, impl_->creation_requirements, api_version(), surface, true, impl_->capabilities));
  impl_->physical_device = std::move(selected_physical);
  impl_->synchronization = detail::select_synchronization_backend(impl_->capabilities);


  VKEXEC_TRY(build_device_into(impl_->physical_device, impl_->device));
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

context::~context() = default;

detail::context_state::~context_state()
{
  // Agents first so outstanding completions finish before tearing down Vulkan objects.
  host_agent.reset();
  completion_waiter.reset();

  if (device.device != VK_NULL_HANDLE) {
    if (owns_device) { vkDeviceWaitIdle(device.device); }
    for (auto const &[cmd, pool] : command_buffer_pools) {
      (void)cmd;
      vkDestroyCommandPool(device.device, pool, nullptr);
    }
    for (auto const &[family, pool] : available_command_pools) {
      (void)family;
      vkDestroyCommandPool(device.device, pool, nullptr);
    }
    if (owns_device) {
      vkb::destroy_device(device);
      has_device = false;
    }
  }

  if (owns_instance && has_instance) {
    vkb::destroy_instance(instance);
    has_instance = false;
  }
}

auto context::lock_queue(VkQueue queue) const -> queue_guard
{ return queue_guard{ impl_->queue_synchronization.state(queue) }; }

auto detail::descriptor_pool_access::lock(context const &ctx, VkDescriptorPool pool) -> std::unique_lock<std::mutex>
{ return std::unique_lock{ *ctx.impl_->descriptor_pool_synchronization.state(pool) }; }

auto detail::descriptor_pool_access::register_owned(context const &ctx, VkDescriptorPool pool) -> status
{
  if (pool == VK_NULL_HANDLE) { return {}; }
#if VKEXEC_HAS_EXCEPTIONS
  try {
#endif
    (void)ctx.impl_->descriptor_pool_synchronization.state(pool);
#if VKEXEC_HAS_EXCEPTIONS
  } catch (...) {
    return fail(unexpected_exception_error());
  }
#endif
  return {};
}

auto context::acquire_command_pool(std::uint32_t family) -> result<VkCommandPool>
{
  {
    std::scoped_lock const lock(impl_->command_pool_cache_mutex);
    auto found = std::ranges::find_if(
      impl_->available_command_pools, [family](auto const &entry) -> bool { return entry.first == family; });
    if (found != impl_->available_command_pools.end()) {
      VkCommandPool pool = found->second;
      impl_->available_command_pools.erase(found);
      return pool;
    }
  }
  VkCommandPoolCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  info.queueFamilyIndex = family;
  info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandPool pool{ VK_NULL_HANDLE };
  if (VkResult const result = vkCreateCommandPool(device(), &info, nullptr, &pool); result != VK_SUCCESS) {
    return fail(result, "vkCreateCommandPool failed");
  }
  return pool;
}

auto context::release_command_pool(std::uint32_t family, VkCommandPool pool) noexcept -> void
{
  if (pool == VK_NULL_HANDLE) { return; }
  if (vkResetCommandPool(device(), pool, 0) != VK_SUCCESS) {
    vkDestroyCommandPool(device(), pool, nullptr);
    return;
  }
#if VKEXEC_HAS_EXCEPTIONS
  try {
#endif
    std::scoped_lock const lock(impl_->command_pool_cache_mutex);
    impl_->available_command_pools.emplace_back(family, pool);
#if VKEXEC_HAS_EXCEPTIONS
  } catch (...) {
    vkDestroyCommandPool(device(), pool, nullptr);
  }
#endif
}

auto context::host_agent_thread_id() -> std::thread::id
{
  if (!impl_->host_agent) { impl_->host_agent = std::make_unique<detail::host_agent>(); }
  return impl_->host_agent->thread_id();
}

auto context::do_enqueue_fence_wait(VkSemaphore semaphore,
  VkFence fence,
  detail::stop_fn stop_requested,
  detail::done_fn on_done) -> status
{
  if (!impl_->completion_waiter && impl_->device.device != VK_NULL_HANDLE) {
    impl_->completion_waiter = std::make_unique<detail::completion_waiter>(
      impl_->device.device, impl_->compute_queue, impl_->queue_synchronization.state(impl_->compute_queue));
  }
  if (!impl_->completion_waiter) {
    error failure = make_error(errc::invalid_argument, "completion waiter requires a VkDevice");
    status result = fail(failure);
    // From here on, ownership of the sync objects is consumed and no error/status
    // construction is allowed to occur before completion is delivered.
    if (fence != VK_NULL_HANDLE) {
      (void)vkWaitForFences(device(), 1, &fence, VK_TRUE, UINT64_MAX);
    } else if (compute_queue() != VK_NULL_HANDLE) {
      auto const lock = lock_queue(compute_queue());
      (void)vkQueueWaitIdle(compute_queue());
    }
    if (semaphore != VK_NULL_HANDLE) { vkDestroySemaphore(device(), semaphore, nullptr); }
    if (fence != VK_NULL_HANDLE) { vkDestroyFence(device(), fence, nullptr); }
    if (on_done) { on_done(std::move(failure), false); }
    return result;
  }
  return impl_->completion_waiter->enqueue(semaphore, fence, std::move(stop_requested), std::move(on_done));
}

auto context::do_enqueue_borrowed_fence_wait(VkFence fence, detail::stop_fn stop_requested, detail::done_fn on_done)
  -> status
{
  if (!impl_->completion_waiter && impl_->device.device != VK_NULL_HANDLE) {
    impl_->completion_waiter = std::make_unique<detail::completion_waiter>(
      impl_->device.device, impl_->compute_queue, impl_->queue_synchronization.state(impl_->compute_queue));
  }
  if (!impl_->completion_waiter) {
    error failure = make_error(errc::invalid_argument, "completion waiter requires a VkDevice");
    status result = fail(failure);
    if (on_done) { on_done(std::move(failure), false); }
    return result;
  }
  return impl_->completion_waiter->enqueue_borrowed(fence, std::move(stop_requested), std::move(on_done));
}

auto context::do_enqueue_host(detail::host_task_fn task) -> status
{
  if (!impl_->host_agent) { impl_->host_agent = std::make_unique<detail::host_agent>(); }
  return impl_->host_agent->enqueue(std::move(task));
}

auto detail::enqueue_host(context_handle const &state, host_task_fn task) -> status
{
  if (!state) { return fail(errc::invalid_argument, "scheduler has no context"); }
  auto facade = context_access::facade(state);
  return facade.enqueue_host(std::move(task));
}

auto detail::retirement_future(context_handle const &state) -> std::shared_future<void>
{ return state != nullptr ? state->retirement : std::shared_future<void>{}; }

auto context::allocate_command_buffer() -> result<VkCommandBuffer>
{
  VKEXEC_TRY_ASSIGN(pool, acquire_command_pool(queue_family()));
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = pool;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  VkCommandBuffer cmd{ VK_NULL_HANDLE };
  if (VkResult const result = vkAllocateCommandBuffers(impl_->device.device, &alloc_info, &cmd); result != VK_SUCCESS) {
    release_command_pool(queue_family(), pool);
    return fail(result, "vkAllocateCommandBuffers failed");
  }
#if VKEXEC_HAS_EXCEPTIONS
  try {
#endif
    std::scoped_lock const lock(impl_->command_buffer_mutex);
    impl_->command_buffer_pools.emplace(cmd, pool);
#if VKEXEC_HAS_EXCEPTIONS
  } catch (...) {
    vkFreeCommandBuffers(device(), pool, 1, &cmd);
    release_command_pool(queue_family(), pool);
    return fail(unexpected_exception_error());
  }
#endif
  return cmd;
}

auto context::free_command_buffer(VkCommandBuffer cmd) -> void
{
  if (cmd == VK_NULL_HANDLE) { return; }
  VkCommandPool pool{ VK_NULL_HANDLE };
  {
    std::scoped_lock const lock(impl_->command_buffer_mutex);
    auto found = impl_->command_buffer_pools.find(cmd);
    if (found == impl_->command_buffer_pools.end()) { return; }
    pool = found->second;
    impl_->command_buffer_pools.erase(found);
  }
  vkFreeCommandBuffers(device(), pool, 1, &cmd);
  release_command_pool(queue_family(), pool);
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

  std::array<VkCommandBuffer, 1> const commands{ cmd };
  auto submitted = submit(queue_submit{ .command_buffers = commands, .fence = fence });
  if (!submitted) {
    vkDestroyFence(impl_->device.device, fence, nullptr);
    return submitted;
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

  std::array<VkCommandBuffer, 1> const commands{ cmd };
  std::array<semaphore_submit, 1> const signals{ semaphore_submit{ .semaphore = sem } };
  auto submitted = submit(queue_submit{ .command_buffers = commands, .signals = signals, .fence = fence });
  if (!submitted) {
    vkDestroySemaphore(impl_->device.device, sem, nullptr);
    if (fence != VK_NULL_HANDLE) { vkDestroyFence(impl_->device.device, fence, nullptr); }
    if (out_fence != nullptr) { *out_fence = VK_NULL_HANDLE; }
    return submitted;
  }
  *out_semaphore = sem;
  return {};
}

auto context::submit(queue_submit const &info) const -> status
{
  auto *queue = info.queue != VK_NULL_HANDLE ? info.queue : impl_->compute_queue;
  return detail::submit(info,
    queue,
    impl_->synchronization,
    impl_->capabilities.timeline_semaphore,
    impl_->procs,
    *impl_->queue_synchronization.state(queue));
}

}// namespace vkexec
