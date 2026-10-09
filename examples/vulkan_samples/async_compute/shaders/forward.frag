#version 450
/* Copyright (c) 2019-2025, Arm Limited and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Adapted from KhronosGroup/Vulkan-Samples async_compute for vkexec descriptors.

layout(set = 0, binding = 0) uniform texture2D baseColorImage;
layout(set = 0, binding = 1) uniform sampler materialSampler;
layout(set = 0, binding = 3) uniform texture2D shadowImage;
layout(set = 0, binding = 4, std430) readonly buffer Light {
  mat4 matrix;
  vec4 direction;
  vec4 color;
} light;
layout(location = 0) in vec3 vColor;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vTexcoord;
layout(location = 3) in vec4 vShadowClip;
layout(location = 0) out vec4 outColor;
void main() {
  vec4 albedo = texture(sampler2D(baseColorImage, materialSampler), vTexcoord);
  vec3 shadowCoord = vShadowClip.xyz / vShadowClip.w;
  vec2 shadowUV = shadowCoord.xy * 0.5 + 0.5;
  float visibility = 1.0;
  if (all(greaterThanEqual(shadowUV, vec2(0.0))) &&
      all(lessThanEqual(shadowUV, vec2(1.0))) && shadowCoord.z <= 1.0) {
    float storedDepth = texture(sampler2D(shadowImage, materialSampler), shadowUV).r;
    visibility = shadowCoord.z <= storedDepth + 0.002 ? 1.0 : 0.2;
  }
  float diffuse = max(dot(normalize(vNormal), -normalize(light.direction.xyz)), 0.0);
  vec3 color = albedo.rgb * vColor * (vec3(0.18) + light.color.rgb * diffuse * visibility);
  outColor = vec4(color, 1.0);
}
