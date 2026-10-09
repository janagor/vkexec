#ifndef VKEXEC_EXAMPLES_FREE_CAMERA_HPP
#define VKEXEC_EXAMPLES_FREE_CAMERA_HPP

#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace vkexec::examples {

inline constexpr float k_default_camera_far_plane = 4000.0F;

struct camera_data
{
  std::array<float, 4> position{};
  std::array<float, 4> right{};
  std::array<float, 4> up{};
  std::array<float, 4> forward{};
  std::array<float, 4> projection{};
};

struct free_camera
{
  std::array<float, 3> position{};
  float yaw{ 0.0F };
  float pitch{ 0.0F };
  float speed{ 1.0F };
  float far_plane{ k_default_camera_far_plane };
  double previous_time{ 0.0 };
  double previous_x{ 0.0 };
  double previous_y{ 0.0 };
  bool dragging{ false };

  free_camera(std::array<float, 3> initial_position,
    std::array<float, 4> rotation,
    float movement_speed,
    float far_distance = k_default_camera_far_plane)
    : position(initial_position), speed(movement_speed), far_plane(far_distance)
  {
    float const rotation_x = rotation.at(0);
    float const rotation_y = rotation.at(1);
    float const rotation_z = rotation.at(2);
    float const rotation_w = rotation.at(3);
    float const forward_x = -2.0F * ((rotation_x * rotation_z) + (rotation_w * rotation_y));
    float const forward_y = -2.0F * ((rotation_y * rotation_z) - (rotation_w * rotation_x));
    float const forward_z = -(1.0F - (2.0F * ((rotation_x * rotation_x) + (rotation_y * rotation_y))));
    yaw = std::atan2(forward_x, -forward_z);
    pitch = std::asin(std::clamp(forward_y, -1.0F, 1.0F));
  }

  auto update(GLFWwindow *window) -> void
  {
    constexpr float k_mouse_sensitivity = 0.002F;
    constexpr float k_max_pitch = 1.55F;
    double const now = glfwGetTime();
    float const delta = static_cast<float>(std::clamp(now - previous_time, 0.0, 0.1));
    previous_time = now;
    double cursor_x = 0.0;
    double cursor_y = 0.0;
    glfwGetCursorPos(window, &cursor_x, &cursor_y);
    bool const pressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (pressed && dragging) {
      yaw += static_cast<float>(cursor_x - previous_x) * k_mouse_sensitivity;
      pitch = std::clamp(
        pitch - (static_cast<float>(cursor_y - previous_y) * k_mouse_sensitivity), -k_max_pitch, k_max_pitch);
    }
    dragging = pressed;
    previous_x = cursor_x;
    previous_y = cursor_y;
    float const sine = std::sin(yaw);
    float const cosine = std::cos(yaw);
    std::array<float, 3> const forward{ std::cos(pitch) * sine, std::sin(pitch), -std::cos(pitch) * cosine };
    std::array<float, 3> const right{ cosine, 0.0F, sine };
    auto move = [&](int key, std::array<float, 3> const &direction, float sign) -> void {
      if (glfwGetKey(window, key) != GLFW_PRESS) { return; }
      for (std::size_t index = 0; index < position.size(); ++index) {
        position.at(index) += direction.at(index) * sign * speed * delta;
      }
    };
    move(GLFW_KEY_W, forward, 1.0F);
    move(GLFW_KEY_S, forward, -1.0F);
    move(GLFW_KEY_D, right, 1.0F);
    move(GLFW_KEY_A, right, -1.0F);
    move(GLFW_KEY_E, { 0.0F, 1.0F, 0.0F }, 1.0F);
    move(GLFW_KEY_Q, { 0.0F, 1.0F, 0.0F }, -1.0F);
  }

  [[nodiscard]] auto data(VkExtent2D extent) const -> camera_data
  {
    float const sine = std::sin(yaw);
    float const cosine = std::cos(yaw);
    float const pitch_sine = std::sin(pitch);
    float const pitch_cosine = std::cos(pitch);
    return camera_data{
      .position = { position.at(0), position.at(1), position.at(2), 0.0F },
      .right = { cosine, 0.0F, sine, 0.0F },
      .up = { -pitch_sine * sine, pitch_cosine, pitch_sine * cosine, 0.0F },
      .forward = { pitch_cosine * sine, pitch_sine, -pitch_cosine * cosine, 0.0F },
      .projection = { 1.0F, far_plane, static_cast<float>(extent.width) / static_cast<float>(extent.height), 0.0F },
    };
  }
};

}// namespace vkexec::examples

#endif
