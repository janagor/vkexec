#version 450

layout(set = 0, binding = 0) uniform texture2D albedoImage;
layout(set = 0, binding = 1) uniform texture2D normalImage;
layout(set = 0, binding = 2) uniform texture2D depthImage;
layout(set = 0, binding = 3) uniform sampler linearSampler;
layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 outColor;

void main() {
  float depth = texture(sampler2D(depthImage, linearSampler), vUv).r;
  if (depth >= 1.0) {
    outColor = vec4(0.02, 0.025, 0.04, 1.0);
    return;
  }
  vec3 albedo = texture(sampler2D(albedoImage, linearSampler), vUv).rgb;
  vec3 normal = normalize(texture(sampler2D(normalImage, linearSampler), vUv).rgb * 2.0 - 1.0);
  vec3 lightDirection = normalize(vec3(0.4, 0.5, 0.8));
  float light = max(dot(normal, lightDirection), 0.0);
  outColor = vec4(albedo * (0.2 + 0.8 * light), 1.0);
}
