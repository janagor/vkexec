#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vkexec {

/// Compile GLSL compute shader source to SPIR-V words via glslang.
std::vector<std::uint32_t> compile_glsl_to_spirv(std::string_view glsl_source, std::string_view name = "vkexec");

} // namespace vkexec
