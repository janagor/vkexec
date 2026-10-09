#version 450
/* Copyright (c) 2021, Arm Limited and Contributors
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

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform texture2D hdrImage;
layout(set = 0, binding = 1) uniform texture2D bloomImage;
layout(set = 0, binding = 2) uniform sampler imageSampler;
void main() {
  vec3 hdr = texture(sampler2D(hdrImage, imageSampler), vUV).rgb;
  vec3 bloom = texture(sampler2D(bloomImage, imageSampler), vUV).rgb;
  vec3 color = hdr + bloom;
  outColor = vec4(color / (1.0 + color), 1.0);
}
