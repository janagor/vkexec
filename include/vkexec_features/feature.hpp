#ifndef VKEXEC_FEATURES_FEATURE_HPP
#define VKEXEC_FEATURES_FEATURE_HPP

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <string_view>

namespace vkexec::feat {

struct timeline_semaphore;
struct buffer_device_address;
struct dynamic_rendering;

template<typename Tag>
struct feature_traits;

template<typename Tag>
concept feature = requires(context const &ctx, vulkan_requirements &req) {
  { feature_traits<Tag>::name() } -> std::convertible_to<std::string_view>;
  { feature_traits<Tag>::available(ctx) } -> std::same_as<bool>;
  { feature_traits<Tag>::configure(req) } -> std::same_as<void>;
};

template<feature Tag>
[[nodiscard]] constexpr auto name() -> std::string_view
{ return feature_traits<Tag>::name(); }

template<feature Tag>
[[nodiscard]] auto available(context const &ctx) -> bool
{ return feature_traits<Tag>::available(ctx); }

template<feature Tag>
auto configure(vulkan_requirements &req) -> void
{ feature_traits<Tag>::configure(req); }

}// namespace vkexec::feat

#endif// VKEXEC_FEATURES_FEATURE_HPP
