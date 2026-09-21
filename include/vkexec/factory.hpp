#ifndef VKEXEC_FACTORY_HPP
#define VKEXEC_FACTORY_HPP

//! \file
//! Core sender factories under `vkexec::factory`.
//!
//! Prefer this when you only need owning create/allocate entry points. Product
//! headers also declare their factories; this umbrella pulls the core set.

#include <vkexec/buffer.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/gpu_buffer.hpp>
#include <vkexec/image.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/sampler.hpp>
#include <vkexec/tensor.hpp>

namespace vkexec {}// namespace vkexec

#endif// VKEXEC_FACTORY_HPP
