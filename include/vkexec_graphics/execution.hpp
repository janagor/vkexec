#ifndef VKEXEC_GRAPHICS_EXECUTION_HPP
#define VKEXEC_GRAPHICS_EXECUTION_HPP

//! \file
//! Layer 1 umbrella: borrowable graphics pipeline handles, record helpers, draw adaptors.
//!
//! Prefer this for embedders that already own buffers / present via `window`.
//! Owning RAII types live in `<vkexec_graphics/resources.hpp>`.

#include <vkexec_graphics/draw.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>
#include <vkexec_graphics/swapchain.hpp>

namespace vkexec {
}// namespace vkexec

#endif// VKEXEC_GRAPHICS_EXECUTION_HPP
