#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_RESOURCE_TABLE_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_RESOURCE_TABLE_HPP

//! \file
//! Descriptor-heap lowering state for core `resource_table` values.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace vkexec {

struct heap_index_binding
{
  std::uint32_t slot{ 0 };
  std::uint32_t index{ 0 };
};

//! Logical resource slot to physical descriptor-heap index mapping.
class heap_index_map
{
public:
  heap_index_map() = default;
  explicit heap_index_map(std::vector<heap_index_binding> entries) : entries_(std::move(entries)) {}

  [[nodiscard]] auto entries() const noexcept -> std::span<heap_index_binding const> { return entries_; }
  [[nodiscard]] auto size() const noexcept -> std::size_t { return entries_.size(); }
  [[nodiscard]] auto index_for(std::uint32_t slot) const noexcept -> std::optional<std::uint32_t>;

private:
  std::vector<heap_index_binding> entries_;
};

//! Extension-only state needed to lower storage buffers into mapped heap slots.
struct heap_table_lower_env
{
  std::span<std::byte> resource_heap_bytes;
  std::size_t buffer_descriptor_size{ 0 };
  std::size_t descriptor_stride{ 0 };
  std::span<std::uint32_t const> indices;
};

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_RESOURCE_TABLE_HPP
