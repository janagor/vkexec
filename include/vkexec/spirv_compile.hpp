#ifndef VKEXEC_SPIRV_COMPILE_HPP
#define VKEXEC_SPIRV_COMPILE_HPP

#include <vkexec/result.hpp>
#include <vkexec/error.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <string_view>
#include <vector>

namespace vkexec {

enum class shader_kind : std::uint8_t { compute, vertex, fragment };

/// Compile GLSL source to SPIR-V words via glslang.
auto compile_glsl_to_spirv(std::string_view glsl_source,
  std::string_view name = "vkexec",
  shader_kind kind = shader_kind::compute,
  std::uint32_t vulkan_api_version = VK_API_VERSION_1_0) -> result<std::vector<std::uint32_t>>;

}// namespace vkexec

#endif// VKEXEC_SPIRV_COMPILE_HPP
