#ifndef VKEXEC_GRAPHICS_DRAW_BASIC_HPP
#define VKEXEC_GRAPHICS_DRAW_BASIC_HPP

#include <vkexec/config.hpp>
#include <vkexec/detail/stop.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/result.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec_graphics/graphics.hpp>
#include <vkexec_graphics/graphics_pipeline_resources.hpp>
#include <vkexec_graphics/mesh.hpp>
#include <vkexec_graphics/presenter.hpp>
#include <vkexec_graphics/submit.hpp>

#include <stdexec/execution.hpp>
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec::detail {

[[nodiscard]] inline auto try_begin_frame(owned::presenter &win) -> result<std::optional<frame>>
{ return win.begin_frame(); }

[[nodiscard]] inline auto try_end_frame(owned::presenter &win, frame const &drawn) -> result<VkFence>
{ return win.end_frame(drawn); }

template<class Receiver>
auto complete_draw(Receiver &&receiver, std::optional<error> failure, bool stopped) noexcept -> void
{
  if (failure) {
    stdexec::set_error(std::forward<Receiver>(receiver), std::move(*failure));
  } else if (stopped) {
    stdexec::set_stopped(std::forward<Receiver>(receiver));
  } else {
    stdexec::set_value(std::forward<Receiver>(receiver));
  }
}

template<class Receiver, class Operation>
auto start_draw_sync(context_handle const &state, Receiver &receiver, Operation &&operation) noexcept -> void
{
#if VKEXEC_HAS_EXCEPTIONS
  try {
#endif
    if (receiver_stop_requested(receiver)) {
      stdexec::set_stopped(std::move(receiver));
      return;
    }
    if (!state) {
      stdexec::set_error(std::move(receiver), make_error(errc::invalid_argument, "draw requires a context"));
      return;
    }
    std::forward<Operation>(operation)();
#if VKEXEC_HAS_EXCEPTIONS
  } catch (...) {
    stdexec::set_error(std::move(receiver), unexpected_exception_error());
  }
#endif
}

template<class WindowOp, class Receiver>
auto start_draw_async(context_handle const &state,
  owned::presenter *win,
  WindowOp &&record_and_end,
  Receiver &receiver) noexcept -> void
{
#if VKEXEC_HAS_EXCEPTIONS
  try {
#endif
    auto const token = receiver_stop_token(receiver);
    if (stop_requested(token)) {
      stdexec::set_stopped(std::move(receiver));
      return;
    }
    if (!state) {
      stdexec::set_error(std::move(receiver), make_error(errc::invalid_argument, "draw requires a context"));
      return;
    }

    auto frame_result = try_begin_frame(*win);
    if (!frame_result) {
      stdexec::set_error(std::move(receiver), std::move(frame_result.error()));
      return;
    }
    if (!frame_result->has_value()) {
      stdexec::set_value(std::move(receiver));
      return;
    }

    auto fence_result = std::forward<WindowOp>(record_and_end)(**frame_result);
    if (!fence_result) {
      stdexec::set_error(std::move(receiver), std::move(fence_result.error()));
      return;
    }

    Receiver *const rcvr = &receiver;
    auto facade = context_access::facade(state);
    (void)facade.enqueue_borrowed_fence_wait(
      *fence_result, token, [rcvr](std::optional<error> wait_error, bool stopped) mutable noexcept -> void {
        complete_draw(std::move(*rcvr), std::move(wait_error), stopped);
      });
#if VKEXEC_HAS_EXCEPTIONS
  } catch (...) {
    stdexec::set_error(std::move(receiver), unexpected_exception_error());
  }
#endif
}

}// namespace vkexec::detail

namespace vkexec {

struct draw_closure
{
  owned::presenter *win{ nullptr };
  owned::graphics_pipeline *pipeline{ nullptr };
  std::uint32_t vertex_count{ 0 };
};

struct draw_mesh_closure
{
  owned::presenter *win{ nullptr };
  owned::graphics_pipeline *pipeline{ nullptr };
  mesh_draw const *drawn{ nullptr };
};

//! One layer in a multi-pipeline frame (`draw_layers`).
struct draw_layer
{
  owned::graphics_pipeline *pipeline{ nullptr };
  std::uint32_t vertex_count{ 0 };
};

struct draw_layers_closure
{
  owned::presenter *win{ nullptr };
  std::vector<draw_layer> layers;
};

/**
 * Builds a closure that presents one frame: acquire -> record draw -> submit -> present.
 *
 * Pipe onto a vkexec schedule predecessor: `schedule(ctx) | draw(win, pipe, 3)`.
 *
 * @param win Window owning the swapchain and frame sync.
 * @param pipeline Graphics pipeline compatible with `win.render_pass()`.
 * @param vertex_count Vertex count for a non-indexed draw.
 */
inline auto draw(owned::presenter &win, owned::graphics_pipeline &pipeline, std::uint32_t vertex_count) -> draw_closure
{ return draw_closure{ .win = &win, .pipeline = &pipeline, .vertex_count = vertex_count }; }

//! Builds a mesh-draw closure for one presented frame.
inline auto draw(owned::presenter &win, owned::graphics_pipeline &pipeline, mesh_draw const &drawn) -> draw_mesh_closure
{ return draw_mesh_closure{ .win = &win, .pipeline = &pipeline, .drawn = &drawn }; }

struct draw_bind_closure
{
  owned::presenter *win{ nullptr };
  handles::graphics_pipeline const *resources{ nullptr };
  VkDescriptorSet set{ VK_NULL_HANDLE };
  std::uint32_t vertex_count{ 0 };
};

struct draw_mesh_bind_closure
{
  owned::presenter *win{ nullptr };
  handles::graphics_pipeline const *resources{ nullptr };
  VkDescriptorSet set{ VK_NULL_HANDLE };
  mesh_draw drawn{};
};

//! Present one frame using borrowable pipeline resources + descriptor set.
inline auto draw(owned::presenter &win,
  handles::graphics_pipeline const &resources,
  VkDescriptorSet set,
  std::uint32_t vertex_count) -> draw_bind_closure
{ return draw_bind_closure{ .win = &win, .resources = &resources, .set = set, .vertex_count = vertex_count }; }

//! Present one indexed mesh frame from borrowed handles.
inline auto
  draw(owned::presenter &win, handles::graphics_pipeline const &resources, VkDescriptorSet set, mesh_draw drawn)
    -> draw_mesh_bind_closure
{ return draw_mesh_bind_closure{ .win = &win, .resources = &resources, .set = set, .drawn = drawn }; }

//! Present one indexed mesh frame from `handles::mesh`.
inline auto draw(owned::presenter &win,
  handles::graphics_pipeline const &resources,
  VkDescriptorSet set,
  handles::mesh const &buffers) -> draw_mesh_bind_closure
{ return draw(win, resources, set, make_mesh_draw(buffers)); }

/**
 * Builds a closure that presents one frame using multiple graphics pipelines in a single render pass.
 *
 * @param layers Pipelines and vertex counts drawn in order within one pass.
 */
inline auto draw_layers(owned::presenter &win, std::initializer_list<draw_layer> layers) -> draw_layers_closure
{ return draw_layers_closure{ .win = &win, .layers = std::vector<draw_layer>(layers) }; }

/**
 * Sender that draws one non-indexed frame and blocks until present submit completes.
 *
 * Completes with `set_value` when acquire returns “recreate and retry” (no frame drawn).
 * Use `| vkexec::submit` for non-blocking fence completion.
 *
 * @see draw_async_sender, draw
 */
struct draw_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures =
    stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(error), stdexec::set_stopped_t()>;

  detail::context_handle state;
  owned::presenter *win{ nullptr };
  owned::graphics_pipeline *pipeline{ nullptr };
  std::uint32_t vertex_count{ 0 };

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .state = state }; }

  template<class Receiver> struct op_state
  {
    detail::context_handle state;
    owned::presenter *win{};
    owned::graphics_pipeline *pipeline{};
    std::uint32_t vertex_count{};
    Receiver receiver;

    auto start() noexcept -> void
    {
      detail::start_draw_sync(state, receiver, [this]() -> void {
        auto frame_result = detail::try_begin_frame(*win);
        if (!frame_result) {
          stdexec::set_error(std::move(receiver), std::move(frame_result.error()));
          return;
        }
        if (!frame_result->has_value()) {
          stdexec::set_value(std::move(receiver));
          return;
        }

        frame const &drawn = **frame_result;
        pipeline->record_pass(drawn.command_buffer, win->render_pass(), drawn.framebuffer, drawn.extent, vertex_count);
        if (auto fence_result = detail::try_end_frame(*win, drawn); !fence_result) {
          stdexec::set_error(std::move(receiver), std::move(fence_result.error()));
          return;
        }
        stdexec::set_value(std::move(receiver));
      });
    }
  };

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) const noexcept(std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .state = state,
      .win = win,
      .pipeline = pipeline,
      .vertex_count = vertex_count,
      .receiver = std::move(receiver),
    };
  }
};

/**
 * Like `draw_sender`, but completes asynchronously via the context fence agent.
 *
 * Produced by `draw_sender | vkexec::submit`.
 */
struct draw_async_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures =
    stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(error), stdexec::set_stopped_t()>;

  detail::context_handle state;
  owned::presenter *win{ nullptr };
  owned::graphics_pipeline *pipeline{ nullptr };
  std::uint32_t vertex_count{ 0 };

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .state = state }; }

  explicit draw_async_sender(draw_sender snd)
    : state(std::move(snd.state)), win(snd.win), pipeline(snd.pipeline), vertex_count(snd.vertex_count)
  {}

  template<class Receiver> struct op_state
  {
    detail::context_handle state;
    owned::presenter *win{};
    owned::graphics_pipeline *pipeline{};
    std::uint32_t vertex_count{};
    Receiver receiver;

    auto start() noexcept -> void
    {
      detail::start_draw_async(
        state,
        win,
        [this](frame &drawn) -> result<VkFence> {
          pipeline->record_pass(
            drawn.command_buffer, win->render_pass(), drawn.framebuffer, drawn.extent, vertex_count);
          return detail::try_end_frame(*win, drawn);
        },
        receiver);
    }
  };

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) & noexcept(std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .state = state,
      .win = win,
      .pipeline = pipeline,
      .vertex_count = vertex_count,
      .receiver = std::move(receiver),
    };
  }

  template<class Receiver>
  [[nodiscard]] auto connect(Receiver receiver) && noexcept(std::is_nothrow_move_constructible_v<Receiver>)
    -> op_state<Receiver>
  {
    return op_state<Receiver>{
      .state = state,
      .win = win,
      .pipeline = pipeline,
      .vertex_count = vertex_count,
      .receiver = std::move(receiver),
    };
  }
};

/**
 * Sender that draws multiple layers in one render pass and blocks until present completes.
 *
 * @see draw_layers, draw_layers_async_sender
 */
}// namespace vkexec

#endif// VKEXEC_GRAPHICS_DRAW_BASIC_HPP
