#ifndef VKEXEC_RESOURCE_TABLE_HPP
#define VKEXEC_RESOURCE_TABLE_HPP

//! \file
//! Heap-agnostic logical resource bindings.

#include <vulkan/vulkan_core.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

//! One storage-buffer resource without descriptor-mechanism metadata.
struct resource_ref
{
  VkBuffer buffer{ VK_NULL_HANDLE };
  VkDeviceSize byte_size{ 0 };
};

//! Associates a logical shader slot with a resource.
struct resource_binding
{
  std::uint32_t slot{ 0 };
  resource_ref resource{};
};

//! Ordered bag of logical resource bindings lowered by a descriptor table backend.
class resource_table
{
public:
  resource_table() = default;
  explicit resource_table(std::vector<resource_binding> entries) : entries_(std::move(entries)) {}

  [[nodiscard]] auto entries() const noexcept -> std::span<resource_binding const> { return entries_; }
  [[nodiscard]] auto size() const noexcept -> std::size_t { return entries_.size(); }
  [[nodiscard]] auto empty() const noexcept -> bool { return entries_.empty(); }

private:
  std::vector<resource_binding> entries_;
};

//! Creates a resource table while preserving argument order.
template<class... Bindings>
  requires(std::same_as<std::remove_cvref_t<Bindings>, resource_binding> && ...)
[[nodiscard]] auto bindings(Bindings &&...entries) -> resource_table
{
  std::vector<resource_binding> resources;
  resources.reserve(sizeof...(Bindings));
  (resources.push_back(std::forward<Bindings>(entries)), ...);
  return resource_table{ std::move(resources) };
}

}// namespace vkexec

#endif// VKEXEC_RESOURCE_TABLE_HPP
