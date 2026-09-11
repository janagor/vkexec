#ifndef VKEXEC_GRAPHICS_WINDOW_HPP
#define VKEXEC_GRAPHICS_WINDOW_HPP

//! \file
//! GLFW window with Vulkan swapchain, render pass, and per-frame sync.

#include <vkexec/context.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>
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

/**
 * Per-frame recording handle returned by `window::begin_frame()`.
 *
 * Record into `command_buffer`, then pass the same `frame` to `end_frame`.
 */
struct frame
{
  VkCommandBuffer command_buffer{ VK_NULL_HANDLE };
  VkFramebuffer framebuffer{ VK_NULL_HANDLE };
  VkExtent2D extent{};
  std::uint32_t image_index{ 0 };
};

/**
 * GLFW window with a Vulkan swapchain and render pass for presentation.
 *
 * Drawing (pipelines, meshes, etc.) belongs in the application, not here.
 * Use `begin_frame` / `end_frame`, or the `draw(...)` stdexec adaptors.
 *
 * ~~~~~~~~~~~{.cpp}
 * auto win = vkexec::sync_wait_value(vkexec::window::create({.title = "demo"}));
 * while (!win.should_close()) {
 *   win.poll_events();
 *   // schedule | draw(win, pipeline, 3) | ...
 * }
 * ~~~~~~~~~~~
 *
 * @see graphics_pipeline, draw, swapchain
 */
class window
{
public:
  /**
   * Window / context creation options.
   *
   * `headless` selects `VK_EXT_headless_surface` (no GLFW display). Extra Vulkan
   * requirements are merged into the owned `context`.
   */
  struct config
  {
    std::uint32_t width{ k_default_window_width };
    std::uint32_t height{ k_default_window_height };
    std::string title{ "vkexec" };
    bool validation_layers{ false };
    bool headless{ false };
    vulkan_requirements requirements{};
  };

  /**
   * Creates a GLFW window, Vulkan context with presentation, and swapchain.
   *
   * @param cfg Window size/title and Vulkan options.
   */
  [[nodiscard]] static auto create(config cfg) -> detail::sync_sender_fn<window>;

  /**
   * Creates a swapchain without GLFW or a display (`VK_EXT_headless_surface`).
   *
   * Intended for CI and tests.
   */
  [[nodiscard]] static auto headless(config cfg) -> detail::sync_sender_fn<window>;
  //! Headless window with default config.
  [[nodiscard]] static auto headless() -> detail::sync_sender_fn<window>;

  ~window();

  window(window const &) = delete;
  auto operator=(window const &) -> window & = delete;
  window(window &&other) noexcept;
  auto operator=(window &&other) noexcept -> window &;

  //! Owned Vulkan context used for queues, device, and VMA.
  [[nodiscard]] auto ctx() noexcept -> context & { return *ctx_; }
  //! Const owned Vulkan context.
  [[nodiscard]] auto ctx() const noexcept -> context const & { return *ctx_; }
  //! Presentation surface (GLFW or headless).
  [[nodiscard]] auto surface() const noexcept -> VkSurfaceKHR { return surface_; }

  //! True when the user requested window close (GLFW); false for headless.
  [[nodiscard]] auto should_close() const noexcept -> bool;
  //! Polls GLFW events (no-op when headless).
  auto poll_events() const -> void;
  //! Waits for the device to become idle.
  auto wait_idle() -> void;

  //! Compatible render pass for swapchain framebuffers.
  [[nodiscard]] auto render_pass() const noexcept -> VkRenderPass { return render_pass_; }
  //! Current swapchain extent (empty when no swapchain).
  [[nodiscard]] auto extent() const noexcept -> VkExtent2D { return swapchain_ ? swapchain_->extent() : VkExtent2D{}; }
  //! Current swapchain color format.
  [[nodiscard]] auto swapchain_format() const noexcept -> VkFormat
  { return swapchain_ ? swapchain_->format() : VK_FORMAT_UNDEFINED; }

  //! Borrowed swapchain pointer (valid after `create` / `headless` completes).
  [[nodiscard]] auto borrowed_swapchain() const noexcept -> swapchain const *
  { return swapchain_ ? std::addressof(*swapchain_) : nullptr; }

  /**
   * Acquires the next swapchain image and begins a primary command buffer.
   *
   * Disengaged optional means the swapchain was recreated (caller should retry
   * on the next loop iteration).
   */
  [[nodiscard]] auto begin_frame() -> result<std::optional<frame>>;

  /**
   * Submits the recorded command buffer and presents.
   *
   * The command buffer must already be ended. Returns the per-frame `in_flight`
   * fence signalled by the submit (owned by the window).
   *
   * @param drawn Frame from a successful `begin_frame`.
   */
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
