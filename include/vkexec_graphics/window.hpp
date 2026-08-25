#ifndef VKEXEC_GRAPHICS_WINDOW_HPP
#define VKEXEC_GRAPHICS_WINDOW_HPP

#include <vkexec/context.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

struct GLFWwindow;

namespace vkexec {

constexpr std::uint32_t k_default_window_width = 800;
constexpr std::uint32_t k_default_window_height = 600;

/// Per-frame recording handle returned by `window::begin_frame()`.
struct frame
{
  VkCommandBuffer command_buffer{ VK_NULL_HANDLE };
  VkFramebuffer framebuffer{ VK_NULL_HANDLE };
  VkExtent2D extent{};
  std::uint32_t image_index{ 0 };
};

/// GLFW window with a Vulkan swapchain and render pass for presentation.
/// Drawing (pipelines, meshes, etc.) belongs in the application, not here.
class window
{
public:
  struct config
  {
    std::uint32_t width{ k_default_window_width };
    std::uint32_t height{ k_default_window_height };
    std::string title{ "vkexec" };
    bool validation_layers{ false };
    bool headless{ false };
    vulkan_requirements requirements{};
  };

  explicit window(config cfg);
  window();
  ~window();

  /// Swapchain without GLFW or a display (`VK_EXT_headless_surface`). For CI/tests.
  [[nodiscard]] static auto headless() -> window;
  [[nodiscard]] static auto headless(config cfg) -> window;

  window(window const &) = delete;
  auto operator=(window const &) -> window & = delete;
  window(window &&) = delete;
  auto operator=(window &&) -> window & = delete;

  [[nodiscard]] auto ctx() noexcept -> context & { return *ctx_; }
  [[nodiscard]] auto ctx() const noexcept -> context const & { return *ctx_; }

  [[nodiscard]] auto should_close() const noexcept -> bool;
  auto poll_events() const -> void;
  auto wait_idle() -> void;

  [[nodiscard]] auto render_pass() const noexcept -> VkRenderPass { return render_pass_; }
  [[nodiscard]] auto extent() const noexcept -> VkExtent2D { return swapchain_extent_; }
  [[nodiscard]] auto swapchain_format() const noexcept -> VkFormat { return swapchain_format_; }

  /// Acquire the next swapchain image and begin a primary command buffer.
  /// Returns nullopt if the swapchain was recreated (caller should retry next loop).
  [[nodiscard]] auto begin_frame() -> std::optional<frame>;

  /// Submit the recorded command buffer and present. The command buffer must already be ended.
  /// Returns the per-frame `in_flight` fence signaled by the submit (owned by the window).
  [[nodiscard]] auto end_frame(frame const &drawn) -> VkFence;

private:
  struct frame_sync
  {
    VkSemaphore image_available{ VK_NULL_HANDLE };
    VkFence in_flight{ VK_NULL_HANDLE };
  };

  auto create_surface() -> void;
  auto create_headless_surface() -> void;
  [[nodiscard]] auto framebuffer_size() const -> std::pair<std::uint32_t, std::uint32_t>;
  auto create_swapchain() -> void;
  auto create_image_views() -> void;
  auto create_render_pass() -> void;
  auto create_depth_resources() -> void;
  auto destroy_depth_resources() noexcept -> void;
  auto create_framebuffers() -> void;
  auto create_frame_resources() -> void;
  auto create_swapchain_sync() -> void;
  auto destroy_swapchain_sync() noexcept -> void;
  auto cleanup_swapchain() -> void;
  auto recreate_swapchain() -> void;

  config cfg_;
  bool headless_{ false };
  GLFWwindow *glfw_{ nullptr };
  std::unique_ptr<context> ctx_;
  VkSurfaceKHR surface_{ VK_NULL_HANDLE };

  vkb::Swapchain swapchain_{};
  VkFormat swapchain_format_{ VK_FORMAT_B8G8R8A8_SRGB };
  VkFormat depth_format_{ VK_FORMAT_UNDEFINED };
  VkExtent2D swapchain_extent_{};
  std::vector<VkImage> swapchain_images_;
  std::vector<VkImageView> swapchain_views_;
  std::vector<VkFramebuffer> framebuffers_;

  VkImage depth_image_{ VK_NULL_HANDLE };
  VmaAllocation depth_allocation_{ VK_NULL_HANDLE };
  VkImageView depth_view_{ VK_NULL_HANDLE };

  VkRenderPass render_pass_{ VK_NULL_HANDLE };

  static constexpr int k_frames = 2;
  std::vector<frame_sync> frames_;
  std::vector<VkCommandBuffer> command_buffers_;
  std::vector<VkSemaphore> render_finished_;
  std::vector<VkFence> images_in_flight_;
  std::uint32_t frame_index_{ 0 };
  std::uint32_t current_image_index_{ 0 };
  bool framebuffer_resized_{ false };
  bool frame_open_{ false };

  static auto on_framebuffer_resize(GLFWwindow *win, int width, int height) -> void;
};

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_WINDOW_HPP
