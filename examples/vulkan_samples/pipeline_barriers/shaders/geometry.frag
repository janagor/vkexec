#version 450

layout(set = 0, binding = 0) uniform texture2D baseColorImage;
layout(set = 0, binding = 1) uniform sampler materialSampler;
layout(location = 0) in vec3 vColor;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vTexcoord;
layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;

void main() {
  vec4 baseColor = texture(sampler2D(baseColorImage, materialSampler), vTexcoord);
  if (baseColor.a < 0.5) { discard; }
  outAlbedo = vec4(baseColor.rgb * vColor, 1.0);
  outNormal = vec4(normalize(vNormal) * 0.5 + 0.5, 1.0);
}
