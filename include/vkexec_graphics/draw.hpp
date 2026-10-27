#ifndef VKEXEC_GRAPHICS_DRAW_HPP
#define VKEXEC_GRAPHICS_DRAW_HPP

#include <vkexec_graphics/draw_layers.hpp>

namespace vkexec {

struct draw_bind_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures =
    stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(error), stdexec::set_stopped_t()>;

  detail::context_handle state;
  owned::presenter *win{ nullptr };
  handles::graphics_pipeline const *resources{ nullptr };
  VkDescriptorSet set{ VK_NULL_HANDLE };
  std::uint32_t vertex_count{ 0 };

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .state = state }; }

  template<class Receiver> struct op_state
  {
    detail::context_handle state;
    owned::presenter *win{ nullptr };
    handles::graphics_pipeline const *resources{ nullptr };
    VkDescriptorSet set{ VK_NULL_HANDLE };
    std::uint32_t vertex_count{ 0 };
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
        record_draw_pass(drawn.command_buffer,
          win->render_pass(),
          drawn.framebuffer,
          drawn.extent,
          resources->cfg,
          bind_graphics(*resources, set),
          vertex_count);
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
      .resources = resources,
      .set = set,
      .vertex_count = vertex_count,
      .receiver = std::move(receiver),
    };
  }
};

struct draw_bind_async_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures =
    stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(error), stdexec::set_stopped_t()>;

  detail::context_handle state;
  owned::presenter *win{ nullptr };
  handles::graphics_pipeline const *resources{ nullptr };
  VkDescriptorSet set{ VK_NULL_HANDLE };
  std::uint32_t vertex_count{ 0 };

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .state = state }; }

  explicit draw_bind_async_sender(draw_bind_sender snd)
    : state(std::move(snd.state)), win(snd.win), resources(snd.resources), set(snd.set), vertex_count(snd.vertex_count)
  {}

  template<class Receiver> struct op_state
  {
    detail::context_handle state;
    owned::presenter *win{};
    handles::graphics_pipeline const *resources{};
    VkDescriptorSet set{};
    std::uint32_t vertex_count{};
    Receiver receiver;

    auto start() noexcept -> void
    {
      detail::start_draw_async(
        state,
        win,
        [this](frame &drawn) -> result<VkFence> {
          record_draw_pass(drawn.command_buffer,
            win->render_pass(),
            drawn.framebuffer,
            drawn.extent,
            resources->cfg,
            bind_graphics(*resources, set),
            vertex_count);
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
      .resources = resources,
      .set = set,
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
      .resources = resources,
      .set = set,
      .vertex_count = vertex_count,
      .receiver = std::move(receiver),
    };
  }
};

struct draw_mesh_bind_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures =
    stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(error), stdexec::set_stopped_t()>;

  detail::context_handle state;
  owned::presenter *win{ nullptr };
  handles::graphics_pipeline const *resources{ nullptr };
  VkDescriptorSet set{ VK_NULL_HANDLE };
  mesh_draw drawn{};

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .state = state }; }

  template<class Receiver> struct op_state
  {
    detail::context_handle state;
    owned::presenter *win{ nullptr };
    handles::graphics_pipeline const *resources{ nullptr };
    VkDescriptorSet set{ VK_NULL_HANDLE };
    mesh_draw drawn{};
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

        frame const &drawn_frame = **frame_result;
        record_draw_pass(drawn_frame.command_buffer,
          win->render_pass(),
          drawn_frame.framebuffer,
          drawn_frame.extent,
          resources->cfg,
          bind_graphics(*resources, set),
          drawn);
        if (auto fence_result = detail::try_end_frame(*win, drawn_frame); !fence_result) {
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
      .resources = resources,
      .set = set,
      .drawn = drawn,
      .receiver = std::move(receiver),
    };
  }
};

struct draw_mesh_bind_async_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures =
    stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(error), stdexec::set_stopped_t()>;

  detail::context_handle state;
  owned::presenter *win{ nullptr };
  handles::graphics_pipeline const *resources{ nullptr };
  VkDescriptorSet set{ VK_NULL_HANDLE };
  mesh_draw drawn{};

  [[nodiscard]] auto get_env() const noexcept -> scheduler_env { return scheduler_env{ .state = state }; }

  explicit draw_mesh_bind_async_sender(draw_mesh_bind_sender snd)
    : state(std::move(snd.state)), win(snd.win), resources(snd.resources), set(snd.set), drawn(snd.drawn)
  {}

  template<class Receiver> struct op_state
  {
    detail::context_handle state;
    owned::presenter *win{};
    handles::graphics_pipeline const *resources{};
    VkDescriptorSet set{};
    mesh_draw drawn{};
    Receiver receiver;

    auto start() noexcept -> void
    {
      detail::start_draw_async(
        state,
        win,
        [this](frame &drawn_frame) -> result<VkFence> {
          record_draw_pass(drawn_frame.command_buffer,
            win->render_pass(),
            drawn_frame.framebuffer,
            drawn_frame.extent,
            resources->cfg,
            bind_graphics(*resources, set),
            drawn);
          return detail::try_end_frame(*win, drawn_frame);
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
      .resources = resources,
      .set = set,
      .drawn = drawn,
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
      .resources = resources,
      .set = set,
      .drawn = drawn,
      .receiver = std::move(receiver),
    };
  }
};

inline auto operator|(schedule_sender snd, draw_bind_closure closure) -> draw_bind_sender
{
  return draw_bind_sender{
    .state = std::move(snd.state),
    .win = closure.win,
    .resources = closure.resources,
    .set = closure.set,
    .vertex_count = closure.vertex_count,
  };
}

inline auto operator|(schedule_sender snd, draw_mesh_bind_closure closure) -> draw_mesh_bind_sender
{
  return draw_mesh_bind_sender{
    .state = std::move(snd.state),
    .win = closure.win,
    .resources = closure.resources,
    .set = closure.set,
    .drawn = closure.drawn,
  };
}

[[nodiscard]] inline auto operator|(draw_bind_sender snd, submit_t /*tag*/) -> draw_bind_async_sender
{ return draw_bind_async_sender{ std::move(snd) }; }

[[nodiscard]] inline auto operator|(draw_mesh_bind_sender snd, submit_t /*tag*/) -> draw_mesh_bind_async_sender
{ return draw_mesh_bind_async_sender{ std::move(snd) }; }

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_DRAW_HPP
