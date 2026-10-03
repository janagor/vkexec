#include <vkexec_tools/spirv_compile.hpp>

volatile decltype(&vkexec::compile_glsl_to_spirv) probe = &vkexec::compile_glsl_to_spirv;

auto main() -> int { return probe == nullptr; }
