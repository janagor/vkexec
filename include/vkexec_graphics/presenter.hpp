#ifndef VKEXEC_GRAPHICS_PRESENTER_HPP
#define VKEXEC_GRAPHICS_PRESENTER_HPP

//! \file
//! Backend-neutral Vulkan presentation with swapchain and per-frame sync.

#include <vkexec/context.hpp>
#include <vkexec/detail/move_only_function.hpp>
#include <vkexec/error.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec_graphics/swapchain.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

constexpr std::uint32_t k_default_presenter_width = 800;
constexpr std::uint32_t k_default_presenter_height = 600;

//! Creates a Vulkan surface for an already-created instance.
using surface_factory = std::function<result<VkSurfaceKHR>(VkInstance)>;

namespace graphics {
  //! Move-only depth attachment whose cleanup is supplied by its allocator.
  class depth_attachment
  {
  public:
    depth_attachment() = default;
    template<class Cleanup>
      requires std::is_nothrow_invocable_r_v<void, Cleanup &> && std::is_nothrow_destructible_v<std::decay_t<Cleanup>>
    depth_attachment(VkImage image, VkImageView view, VkFormat format, VkExtent2D extent, Cleanup cleanup)
      : image_(image), view_(view), format_(format), extent_(extent), cleanup_(std::move(cleanup))
    {}

    ~depth_attachment() { reset(); }
    depth_attachment(depth_attachment const &) = delete;
    auto operator=(depth_attachment const &) -> depth_attachment & = delete;
    depth_attachment(depth_attachment &&other) noexcept
      : image_(std::exchange(other.image_, VK_NULL_HANDLE)), view_(std::exchange(other.view_, VK_NULL_HANDLE)),
        format_(std::exchange(other.format_, VK_FORMAT_UNDEFINED)), extent_(std::exchange(other.extent_, {})),
        cleanup_(std::move(other.cleanup_))
    {}
    auto operator=(depth_attachment &&other) noexcept -> depth_attachment &
    {
      if (this == &other) { return *this; }
      reset();
      image_ = std::exchange(other.image_, VK_NULL_HANDLE);
      view_ = std::exchange(other.view_, VK_NULL_HANDLE);
      format_ = std::exchange(other.format_, VK_FORMAT_UNDEFINED);
      extent_ = std::exchange(other.extent_, {});
      cleanup_ = std::move(other.cleanup_);
      return *this;
    }

    [[nodiscard]] auto image() const noexcept -> VkImage { return image_; }
    [[nodiscard]] auto view() const noexcept -> VkImageView { return view_; }
    [[nodiscard]] auto format() const noexcept -> VkFormat { return format_; }
    [[nodiscard]] auto extent() const noexcept -> VkExtent2D { return extent_; }

  private:
    auto reset() noexcept -> void
    {
      if (cleanup_) { cleanup_(); }
      cleanup_ = nullptr;
      image_ = VK_NULL_HANDLE;
      view_ = VK_NULL_HANDLE;
    }

    VkImage image_{ VK_NULL_HANDLE };
    VkImageView view_{ VK_NULL_HANDLE };
    VkFormat format_{ VK_FORMAT_UNDEFINED };
    VkExtent2D extent_{};
    ::vkexec::detail::move_only_function<void()> cleanup_;
  };

  //! Creates an owning depth attachment for a presenter extent.
  using depth_attachment_factory = std::function<result<depth_attachment>(context &, VkExtent2D, VkFormat)>;
}// namespace graphics

using depth_attachment_factory = graphics::depth_attachment_factory;

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

//! Additional GPU waits for the graphics submit that presents a frame.
struct frame_submit_options
{
  std::span<semaphore_submit const> waits;
};

/**
 * Presentation context creation options.
 *
 * `surface_instance_extensions` and `create_surface` are supplied by the
 * application's windowing system. The returned surface is owned by the presenter.
 */
struct presenter_config
{
  std::uint32_t width{ k_default_presenter_width };
  std::uint32_t height{ k_default_presenter_height };
  bool validation_layers{ false };
  std::vector<char const *> surface_instance_extensions;
  surface_factory create_surface;
  depth_attachment_factory create_depth_attachment;
  vulkan_requirements requirements{};
};

namespace owned {
  class presenter;
}// namespace owned

namespace detail {

  struct make_presenter_factory
  {
    presenter_config cfg;

    [[nodiscard]] auto operator()() -> result<owned::presenter>;
  };

  struct make_headless_presenter_factory
  {
    presenter_config cfg;

    [[nodiscard]] auto operator()() -> result<owned::presenter>;
  };

}// namespace detail

namespace factory {

  struct make_presenter_t
  {

    /**
     * Creates a Vulkan context, invokes the surface factory, and creates presentation resources.
     *
     * @param cfg Initial extent, Vulkan options, extensions, and surface factory.
     */
    [[nodiscard]] auto operator()(presenter_config cfg) const
    { return make_sender(::vkexec::detail::make_presenter_factory{ .cfg = std::move(cfg) }); }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_presenter_t make_presenter{};

  /**
   * Creates a swapchain without GLFW or a display (`VK_EXT_headless_surface`).
   *
   * Intended for CI and tests.
   */
  struct make_headless_presenter_t
  {
    [[nodiscard]] auto operator()(presenter_config cfg) const
    { return make_sender(::vkexec::detail::make_headless_presenter_factory{ .cfg = std::move(cfg) }); }
    //! Headless presenter with default config.
    [[nodiscard]] auto operator()() const { return (*this)(presenter_config{}); }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_headless_presenter_t make_headless_presenter{};

}// namespace factory

/**
 * Backend-neutral owner of a Vulkan presentation context and frame resources.
 *
 * Drawing (pipelines, meshes, etc.) belongs in the application, not here.
 * Use `begin_frame` / `end_frame`, or the `draw(...)` stdexec adaptors.
 *
 * @see graphics_pipeline, draw, swapchain, factory::make_presenter, factory::make_headless_presenter
 */
namespace owned {

  class presenter
  {
  public:
    //! @see presenter_config
    using config = presenter_config;

    ~presenter();

    presenter(presenter const &) = delete;
    auto operator=(presenter const &) -> presenter & = delete;
    presenter(presenter &&other) noexcept;
    auto operator=(presenter &&other) noexcept -> presenter &;

    //! Owned Vulkan context used for queues and device procedures.
    [[nodiscard]] auto ctx() noexcept -> context & { return *ctx_; }
    //! Const owned Vulkan context.
    [[nodiscard]] auto ctx() const noexcept -> context const & { return *ctx_; }
    //! Owned presentation surface.
    [[nodiscard]] auto surface() const noexcept -> VkSurfaceKHR { return surface_; }
    //! Waits for the device to become idle. Do not call concurrently with any
    //! queue operation on this device, including direct embedder queue access.
    auto wait_idle() -> void;

    //! Recreates presentation resources, or suspends acquisition for a zero extent.
    //! Do not call concurrently with other device or queue activity.
    auto resize(std::uint32_t width, std::uint32_t height) -> status;
    //! True when presentation is suspended until the application supplies an extent.
    [[nodiscard]] auto needs_resize() const noexcept -> bool { return resize_required_; }

    //! Compatible render pass for swapchain framebuffers.
    [[nodiscard]] auto render_pass() const noexcept -> VkRenderPass { return render_pass_; }
    //! Current swapchain extent (empty when no swapchain).
    [[nodiscard]] auto extent() const noexcept -> VkExtent2D
    { return swapchain_ ? swapchain_->extent() : VkExtent2D{}; }
    //! Current swapchain color format.
    [[nodiscard]] auto swapchain_format() const noexcept -> VkFormat
    { return swapchain_ ? swapchain_->format() : VK_FORMAT_UNDEFINED; }

    //! Borrowed swapchain pointer (valid after `factory::make_presenter` / `factory::make_headless_presenter`
    //! completes).
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
     * @param options Optional `VkPresentInfoKHR::pNext` chain.
     */
    [[nodiscard]] auto end_frame(frame const &drawn, present_options options = {}) -> result<VkFence>;
    //! Submits with caller GPU waits in addition to the swapchain image wait.
    [[nodiscard]] auto end_frame(frame const &drawn, frame_submit_options submit_options, present_options options = {})
      -> result<VkFence>;

  private:
    friend struct detail::make_presenter_factory;
    friend struct detail::make_headless_presenter_factory;

    struct frame_sync
    {
      VkSemaphore image_available{ VK_NULL_HANDLE };
      VkFence in_flight{ VK_NULL_HANDLE };
    };

    presenter() = default;

    auto init(config cfg) -> status;
    auto create_swapchain() -> status;
    auto create_render_pass() -> status;
    //! Owns a depth attachment supplied by `config::create_depth_attachment`.
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

    graphics::depth_attachment depth_;

    VkRenderPass render_pass_{ VK_NULL_HANDLE };

    static constexpr int k_frames = 2;
    std::vector<frame_sync> frames_;
    std::vector<VkCommandPool> command_pools_;
    std::vector<VkCommandBuffer> command_buffers_;
    std::vector<VkSemaphore> render_finished_;
    std::vector<VkFence> images_in_flight_;
    std::uint32_t frame_index_{ 0 };
    std::uint32_t current_image_index_{ 0 };
    bool resize_required_{ false };
    bool suspended_{ false };
    bool frame_open_{ false };
  };

}// namespace owned

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_PRESENTER_HPP
