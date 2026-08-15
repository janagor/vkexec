#ifndef VKEXEC_DETAIL_SPIRV_HPP
#define VKEXEC_DETAIL_SPIRV_HPP

#include <cstdint>
#include <string_view>
#include <vector>

namespace vkexec {

enum class shader_kind : std::uint8_t { compute, vertex, fragment };

/// Compile GLSL source to SPIR-V words via glslang.
auto compile_glsl_to_spirv(std::string_view glsl_source,
  std::string_view name = "vkexec",
  shader_kind kind = shader_kind::compute) -> std::vector<std::uint32_t>;

} // namespace vkexec

#endif // VKEXEC_DETAIL_SPIRV_HPP
