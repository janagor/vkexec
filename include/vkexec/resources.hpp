#ifndef VKEXEC_RESOURCES_HPP
#define VKEXEC_RESOURCES_HPP

//! \file
//! Resources umbrella: owning RAII Vulkan helpers for greenfield apps.
//!
//! Prefer `<vkexec/execution.hpp>` for borrow-first dispatch. This header covers
//! typed buffers, tensors, images, samplers, and owning `compute_pipeline`.

#include <vkexec/buffer.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/gpu_buffer.hpp>
#include <vkexec/image.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/sampler.hpp>
#include <vkexec/tensor.hpp>

namespace vkexec {}// namespace vkexec

#endif// VKEXEC_RESOURCES_HPP
