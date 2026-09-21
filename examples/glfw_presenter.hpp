#ifndef VKEXEC_EXAMPLES_GLFW_PRESENTER_HPP
#define VKEXEC_EXAMPLES_GLFW_PRESENTER_HPP

#include "sync_wait_helpers.hpp"

#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/result.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_graphics/presenter.hpp>
#include <vkexec_graphics/swapchain.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vkexec::examples {

class glfw_presenter
{
public:
  struct config
  {
    std::uint32_t width{ k_default_presenter_width };
    std::uint32_t height{ k_default_presenter_height };
    std::string title{ "vkexec" };
    bool validation_layers{ false };
    vulkan_requirements requirements{};
  };

  [[nodiscard]] static auto create(config cfg) -> glfw_presenter
  {
    glfwSetErrorCallback([](int, char const *description) -> void {
      if (description != nullptr) { std::cerr << std::format("GLFW: {}\n", description); }
    });
    if (glfwInit() != GLFW_TRUE) { fail_check("glfwInit failed"); }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    glfw_presenter created;
    created.window_ =
      glfwCreateWindow(static_cast<int>(cfg.width), static_cast<int>(cfg.height), cfg.title.c_str(), nullptr, nullptr);
    if (created.window_ == nullptr) {
      glfwTerminate();
      fail_check("glfwCreateWindow failed");
    }
    glfwSetWindowUserPointer(created.window_, &created);
    glfwSetFramebufferSizeCallback(created.window_, &glfw_presenter::on_framebuffer_resize);

    std::uint32_t extension_count = 0;
    char const *const *extensions = glfwGetRequiredInstanceExtensions(&extension_count);
    if (extensions == nullptr || extension_count == 0) { fail_check("glfwGetRequiredInstanceExtensions failed"); }
    std::vector<char const *> surface_extensions;
    surface_extensions.reserve(extension_count);
    for (std::uint32_t index = 0; index < extension_count; ++index) {
      surface_extensions.push_back(extensions[index]);// NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }

    GLFWwindow *const native_window = created.window_;
    created.presenter_ = std::make_unique<presenter>(sync_wait_value(factory::presenter({
      .width = cfg.width,
      .height = cfg.height,
      .validation_layers = cfg.validation_layers,
      .surface_instance_extensions = std::move(surface_extensions),
      .create_surface = [native_window](VkInstance instance) -> result<VkSurfaceKHR> {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (VkResult const result = glfwCreateWindowSurface(instance, native_window, nullptr, &surface);
          result != VK_SUCCESS) {
          return fail(result, "glfwCreateWindowSurface failed");
        }
        return surface;
      },
      .requirements = std::move(cfg.requirements),
    })));
    return created;
  }

  ~glfw_presenter()
  {
    presenter_.reset();
    if (window_ != nullptr) {
      glfwDestroyWindow(window_);
      window_ = nullptr;
      glfwTerminate();
    }
  }

  glfw_presenter(glfw_presenter const &) = delete;
  auto operator=(glfw_presenter const &) -> glfw_presenter & = delete;

  glfw_presenter(glfw_presenter &&other) noexcept
    : window_(std::exchange(other.window_, nullptr)), presenter_(std::move(other.presenter_)), resized_(other.resized_)
  {
    if (window_ != nullptr) { glfwSetWindowUserPointer(window_, this); }
  }

  auto operator=(glfw_presenter &&other) noexcept -> glfw_presenter &
  {
    if (this == &other) { return *this; }
    this->~glfw_presenter();
    new (this) glfw_presenter(std::move(other));
    return *this;
  }

  [[nodiscard]] auto should_close() const noexcept -> bool { return glfwWindowShouldClose(window_) == GLFW_TRUE; }

  auto poll_events() -> void
  {
    glfwPollEvents();
    if (!resized_ && !target().needs_resize()) { return; }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    auto resized = target().resize(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
    if (!resized) { abort_with_error(resized.error()); }
    resized_ = false;
  }

  [[nodiscard]] auto target() noexcept -> presenter & { return *presenter_; }
  [[nodiscard]] auto target() const noexcept -> presenter const & { return *presenter_; }
  [[nodiscard]] auto ctx() noexcept -> context & { return target().ctx(); }
  [[nodiscard]] auto render_pass() const noexcept -> VkRenderPass { return target().render_pass(); }
  [[nodiscard]] auto borrowed_swapchain() const noexcept -> swapchain const * { return target().borrowed_swapchain(); }
  [[nodiscard]] auto begin_frame() -> result<std::optional<frame>> { return target().begin_frame(); }
  [[nodiscard]] auto end_frame(frame const &drawn) -> result<VkFence> { return target().end_frame(drawn); }
  auto wait_idle() -> void { target().wait_idle(); }

private:
  glfw_presenter() = default;

  // GLFW fixes the callback signature.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  static auto on_framebuffer_resize(GLFWwindow *window, int width, int height) -> void
  {
    (void)width;
    (void)height;
    auto *self = static_cast<glfw_presenter *>(glfwGetWindowUserPointer(window));
    if (self != nullptr) { self->resized_ = true; }
  }

  GLFWwindow *window_{ nullptr };
  std::unique_ptr<presenter> presenter_;
  bool resized_{ false };
};

}// namespace vkexec::examples

#endif// VKEXEC_EXAMPLES_GLFW_PRESENTER_HPP
