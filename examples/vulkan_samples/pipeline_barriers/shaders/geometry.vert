#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexcoord;
layout(location = 0) out vec3 vColor;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vTexcoord;
layout(set = 0, binding = 2, std430) readonly buffer Camera {
  vec4 position;
  vec4 right;
  vec4 up;
  vec4 forward;
  vec4 projection;
} camera;

void main() {
  vec3 relativePosition = inPosition - camera.position.xyz;
  vec3 cameraSpace = vec3(dot(relativePosition, camera.right.xyz),
    dot(relativePosition, camera.up.xyz), -dot(relativePosition, camera.forward.xyz));
  float nearPlane = camera.projection.x;
  float farPlane = camera.projection.y;
  float verticalScale = 1.0 / tan(0.5);
  float aspectRatio = camera.projection.z;
  gl_Position = vec4(
    cameraSpace.x * verticalScale / aspectRatio,
    -cameraSpace.y * verticalScale,
    (farPlane / (nearPlane - farPlane)) * cameraSpace.z
      - (farPlane * nearPlane / (farPlane - nearPlane)),
    -cameraSpace.z);
  vColor = inColor;
  vNormal = normalize(inNormal);
  vTexcoord = inTexcoord;
}
