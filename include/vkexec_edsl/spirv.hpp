#ifndef VKEXEC_EDSL_SPIRV_HPP
#define VKEXEC_EDSL_SPIRV_HPP

#include <vkexec/error.hpp>

#include <cstdint>
#include <string_view>
#include <vector>

namespace vkexec::edsl {

enum class shader_kind : std::uint8_t { compute, vertex, fragment };

/// Compile GLSL source to SPIR-V words via glslang.
auto compile_glsl_to_spirv(std::string_view glsl_source,
  std::string_view name = "vkexec",
  shader_kind kind = shader_kind::compute) -> vkexec::result<std::vector<std::uint32_t>>;

}// namespace vkexec::edsl

#endif// VKEXEC_EDSL_SPIRV_HPP
