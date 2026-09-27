#ifndef VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_BUFFER_HPP
#define VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_BUFFER_HPP

//! \file
//! Descriptor heap backing storage supplied by a buffer allocator.

#include <vkexec/detail/normalize_errors.hpp>
#include <vkexec/error.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

namespace vkexec {

constexpr VkDeviceSize k_descriptor_heap_buffer_alignment = 4096;

template<class B>
  requires flushable_buffer_resource<B> && device_addressable_buffer_resource<B>
class descriptor_heap_buffer
{
public:
  explicit descriptor_heap_buffer(B resource) noexcept(std::is_nothrow_move_constructible_v<B>)
    : resource_(std::move(resource))
  {}

  descriptor_heap_buffer(descriptor_heap_buffer const &) = delete;
  auto operator=(descriptor_heap_buffer const &) -> descriptor_heap_buffer & = delete;
  descriptor_heap_buffer(descriptor_heap_buffer &&) noexcept(std::is_nothrow_move_constructible_v<B>) = default;
  auto operator=(descriptor_heap_buffer &&) noexcept(std::is_nothrow_move_assignable_v<B>)
    -> descriptor_heap_buffer & = default;
  ~descriptor_heap_buffer() = default;

  [[nodiscard]] auto handle() const noexcept -> VkBuffer { return resource_.handle(); }
  [[nodiscard]] auto size() const noexcept -> VkDeviceSize { return resource_.size(); }
  [[nodiscard]] auto mapped() noexcept -> std::span<std::byte> { return resource_.mapped(); }
  [[nodiscard]] auto mapped() const noexcept -> std::span<std::byte const> { return resource_.mapped(); }
  [[nodiscard]] auto device_address() const -> result<VkDeviceAddress> { return resource_.device_address(); }
  [[nodiscard]] auto flush() -> status { return resource_.flush(); }

private:
  B resource_;
};

namespace factory {
  struct make_descriptor_heap_buffer_t
  {
    template<buffer_allocator A>
      requires flushable_buffer_resource<typename A::buffer_type>
               && device_addressable_buffer_resource<typename A::buffer_type>
    [[nodiscard]] auto operator()(A &allocator, VkDeviceSize size) const
    {
      using resource_type = A::buffer_type;
      auto prepared = make_sender([size]() -> result<VkDeviceSize> {
        if (size == 0) { return fail(errc::invalid_argument, "descriptor heap size must be > 0"); }
        return size;
      });
      auto composed = std::move(prepared) | stdexec::let_value([&allocator](VkDeviceSize &bytes) -> auto {
        return allocate_buffer(allocator,
                 buffer_create_info{
                   .size = bytes,
                   .usage = VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                   .memory = memory_domain::host_visible,
                   .shader_device_address = true,
                   .alignment = k_descriptor_heap_buffer_alignment,
                 })
               | stdexec::let_value([bytes](resource_type &resource) -> auto {
                   return make_sender([&resource, bytes]() -> result<descriptor_heap_buffer<resource_type>> {
                     if (resource.size() < bytes || resource.mapped().size() < bytes) {
                       return fail(errc::unsupported, "descriptor heap allocation is too small or unmapped");
                     }
                     return descriptor_heap_buffer<resource_type>{ std::move(resource) };
                   });
                 });
      });
      return ::vkexec::detail::normalize_errors(std::move(composed));
    }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_descriptor_heap_buffer_t make_descriptor_heap_buffer{};
}// namespace factory

}// namespace vkexec

#endif// VKEXEC_EXTENSIONS_DESCRIPTOR_HEAP_BUFFER_HPP
