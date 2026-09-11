#ifndef VKEXEC_PUSH_HPP
#define VKEXEC_PUSH_HPP

//! \file
//! Host push-constant upload helpers for compute (and graphics) pipelines.

#include <vkexec/pipeline.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <type_traits>

namespace vkexec {

/**
 * Uploads a raw push-constant blob onto `cmd` for `layout`.
 *
 * @param cmd Command buffer in the recording state.
 * @param layout Pipeline layout that declares the push-constant range.
 * @param data Pointer to `bytes` of host data (must not be null when `bytes` > 0).
 * @param bytes Size of the push-constant range to write.
 */
auto upload_push_constants(VkCommandBuffer cmd, VkPipelineLayout layout, void const *data, std::uint32_t bytes) -> void;

//! Uploads trivially copyable `params` as push constants for `layout`.
template<typename T> auto upload_push_constants(VkCommandBuffer cmd, VkPipelineLayout layout, T const &params) -> void
{
  static_assert(std::is_trivially_copyable_v<T>);
  upload_push_constants(cmd, layout, &params, static_cast<std::uint32_t>(sizeof(T)));
}

//! Uploads trivially copyable `params` using `pipe.pipeline_layout`.
template<typename T>
auto upload_push_constants(VkCommandBuffer cmd, pipeline_resources const &pipe, T const &params) -> void
{ upload_push_constants(cmd, pipe.pipeline_layout, params); }

}// namespace vkexec

#endif// VKEXEC_PUSH_HPP
