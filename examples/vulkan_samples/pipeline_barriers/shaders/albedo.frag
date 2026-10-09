#version 450

layout(set = 0, binding = 0) uniform texture2D albedoImage;
layout(set = 0, binding = 3) uniform sampler linearSampler;
layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 outColor;

void main() {
  outColor = vec4(texture(sampler2D(albedoImage, linearSampler), vUv).rgb, 1.0);
}
