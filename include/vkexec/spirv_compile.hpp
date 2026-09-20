#ifndef VKEXEC_SPIRV_COMPILE_HPP
#define VKEXEC_SPIRV_COMPILE_HPP

#include <vkexec/error.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <string_view>
#include <vector>

namespace vkexec {

//! Shader stage kind for GLSL->SPIR-V compilation.
enum class shader_kind : std::uint8_t { compute, vertex, fragment };

/**
 * Compiles GLSL source to SPIR-V words via glslang.
 *
 * @param glsl_source GLSL text for a single shader stage.
 * @param name Debug name used in compiler diagnostics.
 * @param kind Shader stage to compile.
 * @param vulkan_api_version Target Vulkan API version for the SPIR-V environment.
 * @return SPIR-V words on success, or a parse/compile error.
 */
auto compile_glsl_to_spirv(std::string_view glsl_source,
  std::string_view name = "vkexec",
  shader_kind kind = shader_kind::compute,
  std::uint32_t vulkan_api_version = VK_API_VERSION_1_0) -> result<std::vector<std::uint32_t>>;

}// namespace vkexec

#endif// VKEXEC_SPIRV_COMPILE_HPP
