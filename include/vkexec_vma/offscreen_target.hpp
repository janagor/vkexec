#ifndef VKEXEC_VMA_OFFSCREEN_TARGET_HPP
#define VKEXEC_VMA_OFFSCREEN_TARGET_HPP

//! Allocator-backed offscreen color and optional depth attachments.

#include <vkexec/context.hpp>
#include <vkexec/result.hpp>
#include <vkexec_vma/allocator.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace vkexec::vma {

struct offscreen_target_config
{
  VkExtent2D extent{};
  std::vector<VkFormat> color_formats;
  VkFormat depth_format{ VK_FORMAT_UNDEFINED };
  bool sample_depth{ false };
};

class offscreen_target
{
public:
  offscreen_target(offscreen_target const &) = delete;
  auto operator=(offscreen_target const &) -> offscreen_target & = delete;
  offscreen_target(offscreen_target &&) noexcept;
  auto operator=(offscreen_target &&) noexcept -> offscreen_target &;
  ~offscreen_target();

  [[nodiscard]] static auto create(context &ctx, allocator &alloc, offscreen_target_config const &config)
    -> result<offscreen_target>;

  [[nodiscard]] auto render_pass() const noexcept -> VkRenderPass;
  [[nodiscard]] auto framebuffer() const noexcept -> VkFramebuffer;
  [[nodiscard]] auto extent() const noexcept -> VkExtent2D;
  [[nodiscard]] auto color_image(std::size_t index) const -> VkImage;
  [[nodiscard]] auto color_view(std::size_t index) const -> VkImageView;
  [[nodiscard]] auto depth_image() const noexcept -> VkImage;
  [[nodiscard]] auto depth_view() const noexcept -> VkImageView;
  [[nodiscard]] auto clear_values() const -> std::vector<VkClearValue>;

private:
  struct state;
  explicit offscreen_target(std::unique_ptr<state> owned) noexcept;
  std::unique_ptr<state> state_;
};

}// namespace vkexec::vma

#endif// VKEXEC_VMA_OFFSCREEN_TARGET_HPP
