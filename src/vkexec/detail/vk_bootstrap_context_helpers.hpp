#ifndef VKEXEC_DETAIL_VK_BOOTSTRAP_CONTEXT_HELPERS_HPP
#define VKEXEC_DETAIL_VK_BOOTSTRAP_CONTEXT_HELPERS_HPP

#include "vk_bootstrap_context_state.hpp"
#include "vk_bootstrap_feature.hpp"

#include <vkexec/detail/capability_id.hpp>
#include <vkexec/detail/vk_bootstrap_error.hpp>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <iterator>
#include <ranges>
#include <span>
#include <thread>

namespace vkexec::context_bootstrap {

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

[[nodiscard]] inline auto runtime_reaper() -> retirement_executor &
{
  static retirement_executor executor;
  return executor;
}

inline auto enable_capability(device_capabilities &caps, detail::capability_id capability) noexcept -> void
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
  case detail::capability_id::shader_demote_to_helper_invocation:
    caps.shader_demote_to_helper_invocation = true;
    break;
  case detail::capability_id::shader_untyped_pointers:
    caps.shader_untyped_pointers = true;
    break;
  case detail::capability_id::none:
    break;
  }
}

// Raise the request to the library floor so selectors never ask for less than vkexec needs.
inline auto resolve_api_version(vulkan_requirements const &requirements) -> std::uint32_t
{
  auto const requested = VK_MAKE_API_VERSION(0, requirements.api_version_major, requirements.api_version_minor, 0);
  auto const minimum =
    VK_MAKE_API_VERSION(0, vulkan_library::k_min_api_version_major, vulkan_library::k_min_api_version_minor, 0);
  return requested < minimum ? minimum : requested;
}

inline auto append_unique(std::vector<char const *> &dst, std::span<char const *const> src) -> void
{
  // Pointer identity is enough: callers pass string literals / stable extension name macros.
  for (char const *extension : src) {
    if (extension == nullptr) { continue; }
    bool const exists = std::ranges::any_of(
      dst, [extension](char const *item) -> bool { return item != nullptr && std::strcmp(item, extension) == 0; });
    if (!exists) { dst.push_back(extension); }
  }
}

inline auto merge_instance_extensions(vulkan_requirements const &requirements,
  std::span<char const *const> extra_instance_extensions) -> std::vector<char const *>
{
  // Library baselines first, then user extras; never remove required floors.
  std::vector<char const *> merged;
  append_unique(merged, vulkan_library::required_instance_extensions());
  append_unique(merged, requirements.instance_extensions);
  append_unique(merged, extra_instance_extensions);
  return merged;
}

inline auto merge_device_extensions(vulkan_requirements const &requirements, bool want_present)
  -> std::vector<char const *>
{
  std::vector<char const *> merged;
  append_unique(merged, vulkan_library::required_device_extensions());
  if (want_present) { append_unique(merged, vulkan_library::required_presentation_device_extensions()); }
  append_unique(merged, requirements.device_extensions);
  return merged;
}

inline auto merge_features(vulkan_requirements const &requirements) -> VkPhysicalDeviceFeatures
{
  (void)vulkan_library::required_features();
  return requirements.features;
}

inline auto configure_instance_builder(vkb::InstanceBuilder &builder,
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

inline auto configure_device_selector(vkb::PhysicalDeviceSelector &selector,
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

inline auto contains_extension(std::span<char const *const> extensions, char const *name) -> bool
{
  return std::ranges::any_of(extensions,
    [name](char const *extension) -> bool { return extension != nullptr && std::strcmp(extension, name) == 0; });
}

inline auto instance_extension_available(char const *name) -> bool
{
  std::uint32_t count = 0;
  if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS) { return false; }
  std::vector<VkExtensionProperties> extensions(count);
  if (vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()) != VK_SUCCESS) { return false; }
  return std::ranges::any_of(extensions, [name](VkExtensionProperties const &extension) -> bool {
    return std::strcmp(std::data(extension.extensionName), name) == 0;
  });
}

inline auto configure_optional_synchronization2(vulkan_requirements &requirements, std::uint32_t api_version) -> void
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

inline auto apply_optional_device_requests(vkb::PhysicalDevice &physical_device,
  vulkan_requirements const &requirements,
  device_capabilities &caps) -> void
{
  bool khr_extension_enabled{ contains_extension(
    requirements.device_extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) };
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

inline auto build_headless_instance(scheduler_options const &opts, std::uint32_t api_version) -> result<vkb::Instance>
{
  vkb::InstanceBuilder builder{};
  configure_instance_builder(builder, opts, api_version, {});
  builder.set_headless();
  auto const built = builder.build();
  if (!built) { return fail(detail::make_error_from_vkb(built, "vk-bootstrap InstanceBuilder")); }
  return detail::vkb_take(built);
}

inline auto build_instance_with_extensions(scheduler_options const &opts,
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

inline auto select_physical_device(vkb::Instance const &instance,
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

inline auto build_device_into(vkb::PhysicalDevice const &physical_device, vkb::Device &out) -> status
{
  auto const built = vkb::DeviceBuilder{ physical_device }.build();
  if (!built) { return fail(detail::make_error_from_vkb(built, "vk-bootstrap DeviceBuilder")); }
  out = detail::vkb_take(built);
  return {};
}

[[nodiscard]] inline auto make_context_state() -> detail::context_handle
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

}// namespace vkexec::context_bootstrap

#endif// VKEXEC_DETAIL_VK_BOOTSTRAP_CONTEXT_HELPERS_HPP
