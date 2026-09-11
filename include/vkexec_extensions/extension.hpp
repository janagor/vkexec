#ifndef VKEXEC_EXTENSIONS_EXTENSION_HPP
#define VKEXEC_EXTENSIONS_EXTENSION_HPP

//! \file
//! Extension-tag concept and helpers: `name`, `available`, `configure`.

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <string_view>

namespace vkexec::ext {

struct descriptor_heap;

/**
 * Traits specialization point for an extension tag `Tag`.
 *
 * Provide `name()`, `available(context const &)`, and `configure(vulkan_requirements &)`.
 */
template<typename Tag> struct extension_traits;

/**
 * True when `Tag` is a vkexec extension with a complete `extension_traits` specialization.
 */
template<typename Tag>
concept extension = requires(context const &ctx, vulkan_requirements &req) {
  { extension_traits<Tag>::name() } -> std::convertible_to<std::string_view>;
  { extension_traits<Tag>::available(ctx) } -> std::same_as<bool>;
  { extension_traits<Tag>::configure(req) } -> std::same_as<void>;
};

//! Returns the stable string name of extension `Tag`.
template<extension Tag> [[nodiscard]] constexpr auto name() -> std::string_view
{ return extension_traits<Tag>::name(); }

//! Returns whether extension `Tag` is available on `ctx`.
template<extension Tag> [[nodiscard]] auto available(context const &ctx) -> bool
{ return extension_traits<Tag>::available(ctx); }

//! Mutates `req` so context creation enables extension `Tag`.
template<extension Tag> auto configure(vulkan_requirements &req) -> void { extension_traits<Tag>::configure(req); }

}// namespace vkexec::ext

#endif// VKEXEC_EXTENSIONS_EXTENSION_HPP
