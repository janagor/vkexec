#ifndef VKEXEC_BARRIER_HPP
#define VKEXEC_BARRIER_HPP

//! \file
//! Global and resource-scoped barriers, plus pipeable steps for pass graphs.

#include <vkexec/barrier_params.hpp>
#include <vkexec/pass.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <utility>

namespace vkexec {

/** Records a global memory dependency on a recording command buffer. */
[[nodiscard]] auto memory_barrier(context &ctx, VkCommandBuffer cmd, memory_barrier_params const &params) -> status;

/** Records a buffer memory barrier on a recording command buffer. */
[[nodiscard]] auto buffer_barrier(context &ctx, VkCommandBuffer cmd, buffer_barrier_params const &params) -> status;

/** Records an image layout transition or memory barrier on a recording command buffer. */
[[nodiscard]] auto image_barrier(context &ctx, VkCommandBuffer cmd, image_barrier_params const &params) -> status;

/**
 * Pipeable barrier presets for chaining between compute/graphics pass adaptors.
 *
 * Tag types are invoked as callables on a context and command buffer, or used as
 * `| barrier::compute_to_compute()` style steps in a pass graph.
 */
namespace barrier {

  struct buffer_t
  {
    [[nodiscard]] auto operator()(buffer_barrier_params params) const
      -> detail::expr_closure<buffer_t, buffer_barrier_params>
    { return detail::make_expr_closure(*this, params); }
  };

  struct image_t
  {
    [[nodiscard]] auto operator()(image_barrier_params params) const
      -> detail::expr_closure<image_t, image_barrier_params>
    { return detail::make_expr_closure(*this, params); }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr buffer_t buffer{};
  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr image_t image{};

  //! Transfer writes -> compute shader reads/writes.
  struct transfer_to_compute_t
  {
    [[nodiscard]] auto operator()(context &ctx, VkCommandBuffer cmd) const -> status;
    [[nodiscard]] auto operator()() const -> detail::expr_closure<transfer_to_compute_t, detail::empty_data>
    { return detail::make_expr_closure(*this, detail::empty_data{}); }

    template<vkexec_predecessor Sender>
    [[nodiscard]] auto operator()(Sender &&sender) const
      -> detail::sender_expr<transfer_to_compute_t, detail::empty_data, std::decay_t<Sender>>
    { return std::forward<Sender>(sender) | (*this)(); }
  };

  //! Compute -> compute (shader write to shader read/write).
  struct compute_to_compute_t
  {
    [[nodiscard]] auto operator()(context &ctx, VkCommandBuffer cmd) const -> status;
    [[nodiscard]] auto operator()() const -> detail::expr_closure<compute_to_compute_t, detail::empty_data>
    { return detail::make_expr_closure(*this, detail::empty_data{}); }

    template<vkexec_predecessor Sender>
    [[nodiscard]] auto operator()(Sender &&sender) const
      -> detail::sender_expr<compute_to_compute_t, detail::empty_data, std::decay_t<Sender>>
    { return std::forward<Sender>(sender) | (*this)(); }
  };

  //! Compute -> graphics (shader write to vertex/fragment read).
  struct compute_to_graphics_t
  {
    [[nodiscard]] auto operator()(context &ctx, VkCommandBuffer cmd) const -> status;
    [[nodiscard]] auto operator()() const -> detail::expr_closure<compute_to_graphics_t, detail::empty_data>
    { return detail::make_expr_closure(*this, detail::empty_data{}); }

    template<vkexec_predecessor Sender>
    [[nodiscard]] auto operator()(Sender &&sender) const
      -> detail::sender_expr<compute_to_graphics_t, detail::empty_data, std::decay_t<Sender>>
    { return std::forward<Sender>(sender) | (*this)(); }
  };

  //! Graphics -> compute.
  struct graphics_to_compute_t
  {
    [[nodiscard]] auto operator()(context &ctx, VkCommandBuffer cmd) const -> status;
    [[nodiscard]] auto operator()() const -> detail::expr_closure<graphics_to_compute_t, detail::empty_data>
    { return detail::make_expr_closure(*this, detail::empty_data{}); }

    template<vkexec_predecessor Sender>
    [[nodiscard]] auto operator()(Sender &&sender) const
      -> detail::sender_expr<graphics_to_compute_t, detail::empty_data, std::decay_t<Sender>>
    { return std::forward<Sender>(sender) | (*this)(); }
  };

  //! Compute shader write -> compute shader read (read-after-write).
  struct compute_read_t
  {
    [[nodiscard]] auto operator()(context &ctx, VkCommandBuffer cmd) const -> status;
    [[nodiscard]] auto operator()() const -> detail::expr_closure<compute_read_t, detail::empty_data>
    { return detail::make_expr_closure(*this, detail::empty_data{}); }

    template<vkexec_predecessor Sender>
    [[nodiscard]] auto operator()(Sender &&sender) const
      -> detail::sender_expr<compute_read_t, detail::empty_data, std::decay_t<Sender>>
    { return std::forward<Sender>(sender) | (*this)(); }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr transfer_to_compute_t transfer_to_compute{};
  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr compute_to_compute_t compute_to_compute{};
  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr compute_to_graphics_t compute_to_graphics{};
  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr graphics_to_compute_t graphics_to_compute{};
  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr compute_read_t compute_read{};

}// namespace barrier

namespace detail {

  struct buffer_barrier_record
  {
    buffer_barrier_params params;
    [[nodiscard]] auto operator()(context &ctx, VkCommandBuffer cmd) const -> status
    { return vkexec::buffer_barrier(ctx, cmd, params); }
  };

  struct image_barrier_record
  {
    image_barrier_params params;
    [[nodiscard]] auto operator()(context &ctx, VkCommandBuffer cmd) const -> status
    { return vkexec::image_barrier(ctx, cmd, params); }
  };

  template<class Env>
  [[nodiscard]] auto lower_vkexec_pass_step(barrier::buffer_t /*tag*/,
    buffer_barrier_params params,
    Env const & /*env*/) -> barrier_step<buffer_barrier_record>
  { return make_barrier_step(buffer_barrier_record{ params }); }

  template<class Env>
  [[nodiscard]] auto lower_vkexec_pass_step(barrier::image_t /*tag*/, image_barrier_params params, Env const & /*env*/)
    -> barrier_step<image_barrier_record>
  { return make_barrier_step(image_barrier_record{ params }); }

  template<class Tag, class Env>
    requires std::same_as<Tag, barrier::transfer_to_compute_t> || std::same_as<Tag, barrier::compute_to_compute_t>
             || std::same_as<Tag, barrier::compute_to_graphics_t> || std::same_as<Tag, barrier::graphics_to_compute_t>
             || std::same_as<Tag, barrier::compute_read_t>
  [[nodiscard]] auto lower_vkexec_pass_step(Tag tag, empty_data /*data*/, Env const & /*env*/) -> barrier_step<Tag>
  { return make_barrier_step(std::move(tag)); }

}// namespace detail

}// namespace vkexec

#endif// VKEXEC_BARRIER_HPP
