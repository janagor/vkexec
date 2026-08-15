#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace vkexec {

enum class shader_kind { compute, vertex, fragment };

/// Compile GLSL source to SPIR-V words via glslang.
std::vector<std::uint32_t> compile_glsl_to_spirv(std::string_view glsl_source,
  std::string_view name = "vkexec",
  shader_kind kind = shader_kind::compute);

} // namespace vkexec
