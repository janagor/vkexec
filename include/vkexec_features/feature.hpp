#ifndef VKEXEC_FEATURES_FEATURE_HPP
#define VKEXEC_FEATURES_FEATURE_HPP

//! \file
//! Feature-tag concept and helpers: `name`, `available`, `configure`.

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <string_view>

namespace vkexec::feat {

struct timeline_semaphore;
struct buffer_device_address;
struct dynamic_rendering;

/**
 * Traits specialization point for a feature tag `Tag`.
 *
 * Provide `name()`, `available(context const &)`, and `configure(vulkan_requirements &)`.
 */
template<typename Tag> struct feature_traits;

/**
 * True when `Tag` is a vkexec feature with a complete `feature_traits` specialization.
 */
template<typename Tag>
concept feature = requires(context const &ctx, vulkan_requirements &req) {
  { feature_traits<Tag>::name() } -> std::convertible_to<std::string_view>;
  { feature_traits<Tag>::available(ctx) } -> std::same_as<bool>;
  { feature_traits<Tag>::configure(req) } -> std::same_as<void>;
};

//! Returns the stable string name of feature `Tag`.
template<feature Tag> [[nodiscard]] constexpr auto name() -> std::string_view { return feature_traits<Tag>::name(); }

/**
 * Returns whether feature `Tag` is available on `ctx`'s physical device / API version.
 */
template<feature Tag> [[nodiscard]] auto available(context const &ctx) -> bool
{ return feature_traits<Tag>::available(ctx); }

/**
 * Mutates `req` so context creation enables feature `Tag` (core or KHR promotion).
 */
template<feature Tag> auto configure(vulkan_requirements &req) -> void { feature_traits<Tag>::configure(req); }

}// namespace vkexec::feat

#endif// VKEXEC_FEATURES_FEATURE_HPP
