#version 450

layout(set = 0, binding = 0) readonly buffer PosX { float data[]; } pos_x;
layout(set = 0, binding = 1) readonly buffer PosY { float data[]; } pos_y;
layout(location = 0) out vec3 vColor;

void main() {
  uint i = gl_VertexIndex;
  gl_Position = vec4(pos_x.data[i], pos_y.data[i], 0.0, 1.0);
  gl_PointSize = 3.0;
  float tint = float(i % 31u) / 30.0;
  vColor = mix(vec3(1.0, 0.45, 0.15), vec3(0.35, 0.6, 1.0), tint);
}
