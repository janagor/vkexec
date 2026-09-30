#ifndef VKEXEC_ERROR_HELPERS_HPP
#define VKEXEC_ERROR_HELPERS_HPP

//! \file
//! Helpers that turn Vulkan failures into `error` and `unexpected`.

#include <vkexec/config.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan.h>

#include <string_view>

namespace vkexec {

/**
 * Builds an `error` from a failed `VkResult` and a short context string.
 *
 * @param result Failed Vulkan result (negative `VkResult`).
 * @param context Prefix describing the failing call (e.g. `"vkCreateBuffer"`).
 */
[[nodiscard]] auto make_vk_error(VkResult result, std::string_view context) -> error;

//! Returns `unexpected` wrapping `make_vk_error(result, context)`.
[[nodiscard]] inline auto fail(VkResult result, std::string_view context = {}) -> unexpected<error>
{ return fail(make_vk_error(result, context)); }


}// namespace vkexec

#endif// VKEXEC_ERROR_HELPERS_HPP
