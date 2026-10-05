#version 450
/* Copyright (c) 2019-2021, Arm Limited and Contributors
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

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexcoord;
layout(location = 0) out vec3 vColor;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vTexcoord;
layout(location = 3) out vec4 vShadowClip;
layout(set = 0, binding = 2, std430) readonly buffer Camera {
  vec4 position;
  vec4 right;
  vec4 up;
  vec4 forward;
  vec4 projection;
} camera;
layout(set = 0, binding = 4, std430) readonly buffer Light {
  mat4 matrix;
  vec4 direction;
  vec4 color;
} light;
void main() {
  vec3 relativePosition = inPosition - camera.position.xyz;
  vec3 cameraSpace = vec3(dot(relativePosition, camera.right.xyz),
    dot(relativePosition, camera.up.xyz), -dot(relativePosition, camera.forward.xyz));
  float nearPlane = camera.projection.x;
  float farPlane = camera.projection.y;
  float verticalScale = 1.0 / tan(0.5);
  gl_Position = vec4(cameraSpace.x * verticalScale / camera.projection.z,
    -cameraSpace.y * verticalScale,
    (farPlane / (nearPlane - farPlane)) * cameraSpace.z
      - (farPlane * nearPlane / (farPlane - nearPlane)),
    -cameraSpace.z);
  vColor = inColor;
  vNormal = normalize(inNormal);
  vTexcoord = inTexcoord;
  vShadowClip = light.matrix * vec4(inPosition, 1.0);
}
