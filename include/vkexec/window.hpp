#pragma once

#include <vkexec/context.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct GLFWwindow;

namespace vkexec {

/// Per-frame recording handle returned by `window::begin_frame()`.
struct frame {
  VkCommandBuffer command_buffer{ VK_NULL_HANDLE };
  VkFramebuffer framebuffer{ VK_NULL_HANDLE };
  VkExtent2D extent{};
  std::uint32_t image_index{ 0 };
};

/// GLFW window with a Vulkan swapchain and render pass for presentation.
/// Drawing (pipelines, meshes, etc.) belongs in the application, not here.
class window {
public:
  struct config {
    std::uint32_t width{ 800 };
    std::uint32_t height{ 600 };
    std::string title{ "vkexec" };
  };

  explicit window(config cfg);
  window();
  ~window();

  window(const window &) = delete;
  window &operator=(const window &) = delete;

  [[nodiscard]] context &ctx() noexcept { return *ctx_; }
  [[nodiscard]] const context &ctx() const noexcept { return *ctx_; }

  [[nodiscard]] bool should_close() const noexcept;
  void poll_events();
  void wait_idle();

  [[nodiscard]] VkRenderPass render_pass() const noexcept { return render_pass_; }
  [[nodiscard]] VkExtent2D extent() const noexcept { return swapchain_extent_; }
  [[nodiscard]] VkFormat swapchain_format() const noexcept { return swapchain_format_; }

  /// Acquire the next swapchain image and begin a primary command buffer.
  /// Returns nullopt if the swapchain was recreated (caller should retry next loop).
  [[nodiscard]] std::optional<frame> begin_frame();

  /// Submit the recorded command buffer and present. The command buffer must already be ended.
  void end_frame(const frame &frame);

private:
  struct frame_sync {
    VkSemaphore image_available{ VK_NULL_HANDLE };
    VkSemaphore render_finished{ VK_NULL_HANDLE };
    VkFence in_flight{ VK_NULL_HANDLE };
  };

  void create_surface();
  void create_swapchain();
  void create_image_views();
  void create_render_pass();
  void create_framebuffers();
  void create_frame_resources();
  void cleanup_swapchain();
  void recreate_swapchain();

  config cfg_;
  GLFWwindow *glfw_{ nullptr };
  std::unique_ptr<context> ctx_;
  VkSurfaceKHR surface_{ VK_NULL_HANDLE };

  VkSwapchainKHR swapchain_{ VK_NULL_HANDLE };
  VkFormat swapchain_format_{ VK_FORMAT_B8G8R8A8_SRGB };
  VkExtent2D swapchain_extent_{};
  std::vector<VkImage> swapchain_images_;
  std::vector<VkImageView> swapchain_views_;
  std::vector<VkFramebuffer> framebuffers_;

  VkRenderPass render_pass_{ VK_NULL_HANDLE };

  static constexpr int k_frames = 2;
  std::vector<frame_sync> frames_;
  std::vector<VkCommandBuffer> command_buffers_;
  std::uint32_t frame_index_{ 0 };
  std::uint32_t current_image_index_{ 0 };
  bool framebuffer_resized_{ false };
  bool frame_open_{ false };

  static void on_framebuffer_resize(GLFWwindow *win, int width, int height);
};

} // namespace vkexec
