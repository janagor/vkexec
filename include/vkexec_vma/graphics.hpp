#ifndef VKEXEC_VMA_GRAPHICS_HPP
#define VKEXEC_VMA_GRAPHICS_HPP

//! Header-only VMA adapter for the generic presenter depth factory.

#include <vkexec_graphics/depth_attachment.hpp>
#include <vkexec_vma/allocator.hpp>
#include <vkexec_vma/image.hpp>

namespace vkexec::vma {

[[nodiscard]] inline auto make_depth_attachment_factory() -> graphics::depth_attachment_factory
{
  // Presenter constructs its context internally, so each attachment owns an allocator.
  return graphics::make_depth_attachment_factory<allocator>(
    [](context &ctx) -> auto { return factory::make_allocator(ctx); });
}

}// namespace vkexec::vma

#endif// VKEXEC_VMA_GRAPHICS_HPP
