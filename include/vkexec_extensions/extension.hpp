#ifndef VKEXEC_EXTENSIONS_EXTENSION_HPP
#define VKEXEC_EXTENSIONS_EXTENSION_HPP

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <string_view>

namespace vkexec::ext {

struct descriptor_heap;

template<typename Tag>
struct extension_traits;

template<typename Tag>
concept extension = requires(context const &ctx, vulkan_requirements &req) {
  { extension_traits<Tag>::name() } -> std::convertible_to<std::string_view>;
  { extension_traits<Tag>::available(ctx) } -> std::same_as<bool>;
  { extension_traits<Tag>::configure(req) } -> std::same_as<void>;
};

template<extension Tag>
[[nodiscard]] constexpr auto name() -> std::string_view
{ return extension_traits<Tag>::name(); }

template<extension Tag>
[[nodiscard]] auto available(context const &ctx) -> bool
{ return extension_traits<Tag>::available(ctx); }

template<extension Tag>
auto configure(vulkan_requirements &req) -> void
{ extension_traits<Tag>::configure(req); }

}// namespace vkexec::ext

#endif// VKEXEC_EXTENSIONS_EXTENSION_HPP
