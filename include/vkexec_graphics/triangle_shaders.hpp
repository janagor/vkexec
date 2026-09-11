#ifndef VKEXEC_GRAPHICS_TRIANGLE_SHADERS_HPP
#define VKEXEC_GRAPHICS_TRIANGLE_SHADERS_HPP

//! \file
//! Embedded GLSL sources for a simple colored triangle (examples / smoke tests).

#include <string_view>

namespace vkexec::shaders {

//! Vertex shader: hard-coded triangle positions and per-vertex colors.
constexpr std::string_view k_triangle_vert = R"(
#version 450
layout(location = 0) out vec3 vColor;
void main() {
  vec2 pos[3] = vec2[](vec2(0.0, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5));
  vec3 col[3] = vec3[](vec3(1.0, 0.2, 0.2), vec3(0.2, 1.0, 0.2), vec3(0.2, 0.4, 1.0));
  gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
  vColor = col[gl_VertexIndex];
}
)";

//! Fragment shader: outputs interpolated vertex color.
constexpr std::string_view k_triangle_frag = R"(
#version 450
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;
void main() {
  outColor = vec4(vColor, 1.0);
}
)";

}// namespace vkexec::shaders

#endif// VKEXEC_GRAPHICS_TRIANGLE_SHADERS_HPP
