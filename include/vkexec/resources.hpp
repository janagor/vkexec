#ifndef VKEXEC_RESOURCES_HPP
#define VKEXEC_RESOURCES_HPP

//! \file
//! Resources umbrella: allocator-neutral owning Vulkan helpers.
//!
//! Prefer `<vkexec/execution.hpp>` for borrow-first dispatch. This header covers
//! `owned::` samplers and pipelines, plus allocator-generic buffers and tensors.
//! Concrete allocation backends such as VMA live in their own modules.

#include <vkexec/buffer.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/sampler.hpp>
#include <vkexec/tensor.hpp>
#include <vkexec/tensor_pass.hpp>
#include <vkexec/tensor_sync.hpp>

namespace vkexec {}// namespace vkexec

#endif// VKEXEC_RESOURCES_HPP
