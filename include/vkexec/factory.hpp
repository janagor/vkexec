#ifndef VKEXEC_FACTORY_HPP
#define VKEXEC_FACTORY_HPP

//! \file
//! Allocator-neutral sender factories under `vkexec::factory`.
//!
//! Product headers also declare their factories. Buffer and tensor factories
//! accept any allocator modeling the required resource protocol.

#include <vkexec/buffer.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/sampler.hpp>
#include <vkexec/tensor.hpp>

namespace vkexec {}// namespace vkexec

#endif// VKEXEC_FACTORY_HPP
