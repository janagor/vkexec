#ifndef VKEXEC_GRAPHICS_DEPTH_ATTACHMENT_HPP
#define VKEXEC_GRAPHICS_DEPTH_ATTACHMENT_HPP

//! \file
//! Allocator-generic presenter depth attachment creation.

#include <vkexec/detail/normalize_errors.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_graphics/presenter.hpp>

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

namespace vkexec::graphics {

namespace detail {
  template<class F, class A>
  concept allocator_factory =
    std::invocable<F &, context &> && vkexec_sender_of<std::invoke_result_t<F &, context &>, A>;
}// namespace detail

template<image_allocator A, class MakeAllocator>
  requires detail::allocator_factory<MakeAllocator, A> && std::copy_constructible<MakeAllocator>
           && std::is_nothrow_destructible_v<A> && std::is_nothrow_destructible_v<typename A::image_type>
[[nodiscard]] auto make_depth_attachment_factory(MakeAllocator make_allocator) -> depth_attachment_factory
{
  return [make_allocator = std::move(make_allocator)](
           context &ctx, VkExtent2D extent, VkFormat format) -> result<depth_attachment> {
    struct owned_depth
    {
      std::unique_ptr<A> allocator;
      A::image_type image;
      owned::image_view view;
    };
    auto created = make_allocator(ctx) | stdexec::let_value([&ctx, extent, format](A &created_allocator) -> auto {
      auto allocator = std::make_unique<A>(std::move(created_allocator));
      A *const allocation_source = allocator.get();
      return allocate_image(*allocation_source,
               image_create_info{
                 .extent = { .width = extent.width, .height = extent.height, .depth = 1 },
                 .format = format,
                 .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
               })
             | stdexec::let_value(
               [&ctx, extent, format, allocator = std::move(allocator)](A::image_type &image) mutable -> auto {
                 return factory::make_image_view(ctx, image.handle(), format, VK_IMAGE_ASPECT_DEPTH_BIT)
                        | stdexec::let_value([extent, format, &image, allocator = std::move(allocator)](
                                               owned::image_view &view) mutable -> auto {
                            return make_sender(
                              [extent, format, &image, &view, allocator = std::move(allocator)]() mutable
                                -> result<depth_attachment> {
                                VkImage image_handle = image.handle();
                                VkImageView view_handle = view.handle();
                                auto state = std::make_unique<owned_depth>(
                                  owned_depth{ std::move(allocator), std::move(image), std::move(view) });
                                return depth_attachment{ image_handle,
                                  view_handle,
                                  format,
                                  extent,
                                  [state = std::move(state)]() mutable noexcept -> void { state.reset(); } };
                              });
                          });
               });
    });
    // Presenter construction is synchronous; this is its explicit sender-to-result boundary.
    auto normalized = ::vkexec::detail::normalize_errors(std::move(created));
    static_assert(vkexec_sender_of<decltype(normalized), depth_attachment>);
    return try_sync_wait_value(std::move(normalized));
  };
}

}// namespace vkexec::graphics

#endif// VKEXEC_GRAPHICS_DEPTH_ATTACHMENT_HPP
