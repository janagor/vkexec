#ifndef VKEXEC_TOOLS_SPIRV_COMPILE_HPP
#define VKEXEC_TOOLS_SPIRV_COMPILE_HPP

#include <cstdint>
#include <string_view>
#include <vector>
#include <vkexec/result.hpp>
#include <vulkan/vulkan_core.h>

namespace vkexec {

//! Shader stage kind for optional GLSL-to-SPIR-V compilation tools.
enum class shader_kind : std::uint8_t { compute, vertex, fragment };

auto compile_glsl_to_spirv(std::string_view glsl_source,
  std::string_view name = "vkexec",
  shader_kind kind = shader_kind::compute,
  std::uint32_t vulkan_api_version = VK_API_VERSION_1_0) -> result<std::vector<std::uint32_t>>;

}// namespace vkexec

#endif
