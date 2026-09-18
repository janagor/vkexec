#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_STRATEGY_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_STRATEGY_HPP

#include <vkexec/detail/descriptor_backend.hpp>

namespace vkexec {

struct descriptor_heap_t
{
};
// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr descriptor_heap_t descriptor_heap{};

namespace detail {

struct heap_descriptor_backend
{
  [[nodiscard]] static constexpr auto pipeline_create_flags() noexcept -> VkPipelineCreateFlags2
  { return VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT; }

  template<class Resources>
  [[nodiscard]] static auto create_set_and_pipeline_layout(
    VkDevice /*device*/, Resources & /*resources*/, descriptor_layout_info const & /*info*/) -> status
  { return {}; }

  template<class Resources>
  [[nodiscard]] static auto create_descriptor_pool(
    VkDevice /*device*/, Resources & /*resources*/, descriptor_layout_info const & /*info*/) -> status
  { return {}; }

  template<class Resources>
  static auto destroy_binding_objects(VkDevice /*device*/, Resources &resources) -> void
  {
    resources.pipeline_layout = VK_NULL_HANDLE;
    resources.descriptor_pool = VK_NULL_HANDLE;
    resources.set_layout = VK_NULL_HANDLE;
  }

  template<class Bind>
  static auto bind_resources(VkCommandBuffer /*cmd*/, VkPipelineBindPoint /*bind_point*/, Bind const & /*bind*/) -> void
  {}

  template<class Bind>
  [[nodiscard]] static auto push(context const *ctx,
    VkCommandBuffer cmd,
    VkPipelineBindPoint bind_point,
    Bind const & /*bind*/,
    std::span<std::byte const> bytes) -> status
  { return push_bytes(ctx, cmd, bind_point, bytes); }

private:
  [[nodiscard]] static auto push_bytes(context const *ctx,
    VkCommandBuffer cmd,
    VkPipelineBindPoint bind_point,
    std::span<std::byte const> bytes) -> status;
};

static_assert(descriptor_backend<heap_descriptor_backend>);

}// namespace detail
}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_STRATEGY_HPP
