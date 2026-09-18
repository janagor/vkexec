#ifndef VKEXEC_GRAPHICS_PRESENTER_HPP
#define VKEXEC_GRAPHICS_PRESENTER_HPP

//! \file
//! Backend-neutral Vulkan presentation with swapchain and per-frame sync.

#include <vkexec/context.hpp>
#include <vkexec/detail/sync_sender.hpp>
#include <vkexec/error.hpp>
#include <vkexec/result.hpp>
#include <vkexec_graphics/swapchain.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace vkexec {

constexpr std::uint32_t k_default_presenter_width = 800;
constexpr std::uint32_t k_default_presenter_height = 600;

//! Creates a Vulkan surface for an already-created instance.
using surface_factory = std::function<result<VkSurfaceKHR>(VkInstance)>;

/**
 * Per-frame recording handle returned by `presenter::begin_frame()`.
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
 * Backend-neutral owner of a Vulkan presentation context and frame resources.
 *
 * Drawing (pipelines, meshes, etc.) belongs in the application, not here.
 * Use `begin_frame` / `end_frame`, or the `draw(...)` stdexec adaptors.
 *
 * @see graphics_pipeline, draw, swapchain
 */
class presenter
{
public:
  /**
   * Presentation context creation options.
   *
   * `surface_instance_extensions` and `create_surface` are supplied by the
   * application's windowing system. The returned surface is owned by the presenter.
   */
  struct config
  {
    std::uint32_t width{ k_default_presenter_width };
    std::uint32_t height{ k_default_presenter_height };
    bool validation_layers{ false };
    std::vector<char const *> surface_instance_extensions;
    surface_factory create_surface;
    vulkan_requirements requirements{};
  };

  /**
   * Creates a Vulkan context, invokes the surface factory, and creates presentation resources.
   *
   * @param cfg Initial extent, Vulkan options, extensions, and surface factory.
   */
  [[nodiscard]] static auto create(config cfg) -> detail::sync_sender_fn<presenter>;

  /**
   * Creates a swapchain without GLFW or a display (`VK_EXT_headless_surface`).
   *
   * Intended for CI and tests.
   */
  [[nodiscard]] static auto headless(config cfg) -> detail::sync_sender_fn<presenter>;
  //! Headless presenter with default config.
  [[nodiscard]] static auto headless() -> detail::sync_sender_fn<presenter>;

  ~presenter();

  presenter(presenter const &) = delete;
  auto operator=(presenter const &) -> presenter & = delete;
  presenter(presenter &&other) noexcept;
  auto operator=(presenter &&other) noexcept -> presenter &;

  //! Owned Vulkan context used for queues, device, and VMA.
  [[nodiscard]] auto ctx() noexcept -> context & { return *ctx_; }
  //! Const owned Vulkan context.
  [[nodiscard]] auto ctx() const noexcept -> context const & { return *ctx_; }
  //! Owned presentation surface.
  [[nodiscard]] auto surface() const noexcept -> VkSurfaceKHR { return surface_; }
  //! Waits for the device to become idle.
  auto wait_idle() -> void;

  //! Recreates presentation resources, or suspends acquisition for a zero extent.
  auto resize(std::uint32_t width, std::uint32_t height) -> status;
  //! True when presentation is suspended until the application supplies an extent.
  [[nodiscard]] auto needs_resize() const noexcept -> bool { return resize_required_; }

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
   * Disengaged optional means presentation is suspended or requires `resize`.
   */
  [[nodiscard]] auto begin_frame() -> result<std::optional<frame>>;

  /**
   * Submits the recorded command buffer and presents.
   *
   * The command buffer must already be ended. Returns the per-frame `in_flight`
   * fence signalled by the submit (owned by the presenter).
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

  presenter() = default;

  auto init(config cfg) -> status;
  auto create_swapchain() -> status;
  auto create_render_pass() -> status;
  auto create_depth_resources() -> status;
  auto destroy_depth_resources() noexcept -> void;
  auto create_framebuffers() -> status;
  auto create_frame_resources() -> status;
  auto create_swapchain_sync() -> status;
  auto destroy_swapchain_sync() noexcept -> void;
  auto cleanup_swapchain() -> void;
  auto recreate_swapchain(std::uint32_t width, std::uint32_t height) -> status;

  config cfg_;
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
  bool resize_required_{ false };
  bool suspended_{ false };
  bool frame_open_{ false };
};

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_PRESENTER_HPP
