#ifndef VKEXEC_GRAPHICS_WINDOW_HPP
#define VKEXEC_GRAPHICS_WINDOW_HPP

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec_graphics/swapchain.hpp>

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

  [[nodiscard]] static auto create(config cfg) -> result<window>;

  /// Swapchain without GLFW or a display (`VK_EXT_headless_surface`). For CI/tests.
  [[nodiscard]] static auto headless() -> result<window>;
  [[nodiscard]] static auto headless(config cfg) -> result<window>;

  ~window();

  window(window const &) = delete;
  auto operator=(window const &) -> window & = delete;
  window(window &&other) noexcept;
  auto operator=(window &&other) noexcept -> window &;

  [[nodiscard]] auto ctx() noexcept -> context & { return *ctx_; }
  [[nodiscard]] auto ctx() const noexcept -> context const & { return *ctx_; }
  [[nodiscard]] auto surface() const noexcept -> VkSurfaceKHR { return surface_; }

  [[nodiscard]] auto should_close() const noexcept -> bool;
  auto poll_events() const -> void;
  auto wait_idle() -> void;

  [[nodiscard]] auto render_pass() const noexcept -> VkRenderPass { return render_pass_; }
  [[nodiscard]] auto extent() const noexcept -> VkExtent2D { return swapchain_ ? swapchain_->extent() : VkExtent2D{}; }
  [[nodiscard]] auto swapchain_format() const noexcept -> VkFormat
  { return swapchain_ ? swapchain_->format() : VK_FORMAT_UNDEFINED; }

  /// Acquire the next swapchain image and begin a primary command buffer.
  /// Disengaged optional means the swapchain was recreated (caller should retry next loop).
  [[nodiscard]] auto begin_frame() -> result<std::optional<frame>>;

  /// Submit the recorded command buffer and present. The command buffer must already be ended.
  /// Returns the per-frame `in_flight` fence signaled by the submit (owned by the window).
  [[nodiscard]] auto end_frame(frame const &drawn) -> result<VkFence>;

private:
  struct frame_sync
  {
    VkSemaphore image_available{ VK_NULL_HANDLE };
    VkFence in_flight{ VK_NULL_HANDLE };
  };

  window() = default;

  auto init(config cfg) -> status;
  auto create_surface() -> status;
  auto create_headless_surface() -> status;
  [[nodiscard]] auto framebuffer_size() const -> std::pair<std::uint32_t, std::uint32_t>;
  auto create_swapchain() -> status;
  auto create_render_pass() -> status;
  auto create_depth_resources() -> status;
  auto destroy_depth_resources() noexcept -> void;
  auto create_framebuffers() -> status;
  auto create_frame_resources() -> status;
  auto create_swapchain_sync() -> status;
  auto destroy_swapchain_sync() noexcept -> void;
  auto cleanup_swapchain() -> void;
  auto recreate_swapchain() -> status;

  config cfg_;
  bool headless_{ false };
  GLFWwindow *glfw_{ nullptr };
  std::unique_ptr<context> ctx_;
  VkSurfaceKHR surface_{ VK_NULL_HANDLE };

  std::optional<swapchain> swapchain_;
  VkFormat depth_format_{ VK_FORMAT_UNDEFINED };
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
