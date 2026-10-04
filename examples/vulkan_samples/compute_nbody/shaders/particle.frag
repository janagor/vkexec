#version 450

layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;

void main() {
  vec2 circle = gl_PointCoord * 2.0 - 1.0;
  if (dot(circle, circle) > 1.0) { discard; }
  outColor = vec4(vColor, 1.0);
}
