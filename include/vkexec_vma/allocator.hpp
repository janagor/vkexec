#ifndef VKEXEC_VMA_ALLOCATOR_HPP
#define VKEXEC_VMA_ALLOCATOR_HPP

//! \file
//! VMA allocator ownership for resources used with a vkexec context.

#include <vkexec/context.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>

#include <vk_mem_alloc.h>

namespace vkexec::vma {

class allocator;
class gpu_buffer;
class image;

namespace detail {

  struct make_allocator_factory
  {
    context *ctx;

    [[nodiscard]] auto operator()() const -> result<allocator>;
  };

  struct adopt_allocator_factory
  {
    context *ctx;
    VmaAllocator handle;

    [[nodiscard]] auto operator()() const -> result<allocator>;
  };

}// namespace detail

class allocator
{
public:
  using buffer_type = gpu_buffer;
  using image_type = image;

  ~allocator();

  allocator(allocator const &) = delete;
  auto operator=(allocator const &) -> allocator & = delete;

  allocator(allocator &&other) noexcept;
  auto operator=(allocator &&other) noexcept -> allocator &;

  [[nodiscard]] auto native_handle() const noexcept -> VmaAllocator { return handle_; }
  [[nodiscard]] auto ctx() const noexcept -> context & { return *ctx_; }
  [[nodiscard]] auto owns_handle() const noexcept -> bool { return owns_; }

private:
  friend struct detail::make_allocator_factory;
  friend struct detail::adopt_allocator_factory;

  allocator(context *ctx, VmaAllocator handle, bool owns) noexcept : ctx_(ctx), handle_(handle), owns_(owns) {}

  auto destroy() noexcept -> void;

  context *ctx_{ nullptr };
  VmaAllocator handle_{ VK_NULL_HANDLE };
  bool owns_{ false };
};

namespace factory {

  struct make_allocator_t
  {
    [[nodiscard]] auto operator()(context &ctx) const
    { return make_sender(detail::make_allocator_factory{ .ctx = &ctx }); }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_allocator_t make_allocator{};

  struct adopt_allocator_t
  {
    [[nodiscard]] auto operator()(context &ctx, VmaAllocator handle) const
    { return make_sender(detail::adopt_allocator_factory{ .ctx = &ctx, .handle = handle }); }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr adopt_allocator_t adopt_allocator{};

}// namespace factory

}// namespace vkexec::vma

#endif// VKEXEC_VMA_ALLOCATOR_HPP
