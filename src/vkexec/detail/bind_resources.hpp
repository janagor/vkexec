#ifndef VKEXEC_PRIVATE_BIND_RESOURCES_HPP
#define VKEXEC_PRIVATE_BIND_RESOURCES_HPP

#include <vkexec/detail/lower_and_bind_push.hpp>
#include <vkexec/pass.hpp>

#include <memory>
#include <optional>
#include <utility>

namespace vkexec::detail {

template<class Backend> struct bound_release_state
{
  context *ctx{ nullptr };
  pipeline_resources const *pipe{ nullptr };
  std::optional<typename Backend::bound_type> bound;

  bound_release_state() = default;
  bound_release_state(bound_release_state const &) = delete;
  auto operator=(bound_release_state const &) -> bound_release_state & = delete;
  bound_release_state(bound_release_state &&) = delete;
  auto operator=(bound_release_state &&) -> bound_release_state & = delete;

  auto release() noexcept -> void
  {
    if (ctx != nullptr && pipe != nullptr && bound.has_value()) {
      Backend::release(*ctx, *pipe, *bound);
      bound.reset();
    }
  }

  ~bound_release_state() { release(); }
};

template<class Backend, class LowerEnv>
[[nodiscard]] auto make_bind_resources_step(pipeline_resources const *pipe,
  resource_table table,
  LowerEnv env,
  std::vector<std::byte> push) -> pass_step
{
  auto state = std::make_shared<bound_release_state<Backend>>();
  return pass_step{
    .record = [pipe, table = std::move(table), env = std::move(env), push = std::move(push), state](
                context &ctx, VkCommandBuffer cmd, pass_cleanup &) mutable -> status {
      if (pipe == nullptr) { return fail(errc::invalid_argument, "bind_resources requires a pipeline"); }
      state->release();
      auto lowered = lower_and_bind_push<Backend>(ctx, cmd, VK_PIPELINE_BIND_POINT_COMPUTE, *pipe, table, env, push);
      if (!lowered) { return fail(lowered); }
      state->ctx = &ctx;
      state->pipe = pipe;
      state->bound.emplace(expected_take(lowered));
      return {};
    },
    .after_gpu = [state]() -> void { state->release(); },
  };
}

}// namespace vkexec::detail

#endif// VKEXEC_PRIVATE_BIND_RESOURCES_HPP
