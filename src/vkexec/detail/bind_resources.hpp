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
  context_handle state;
  handles::compute_pipeline const *pipe{ nullptr };
  std::optional<typename Backend::bound_type> bound;

  bound_release_state() = default;
  bound_release_state(bound_release_state const &) = delete;
  auto operator=(bound_release_state const &) -> bound_release_state & = delete;
  bound_release_state(bound_release_state &&) = delete;
  auto operator=(bound_release_state &&) -> bound_release_state & = delete;

  auto release() noexcept -> void
  {
    if (state && pipe != nullptr && bound.has_value()) {
      auto facade = context_access::facade(state);
      Backend::release(facade, *pipe, *bound);
      bound.reset();
    }
    state.reset();
    pipe = nullptr;
  }

  ~bound_release_state() { release(); }
};

}// namespace vkexec::detail

#endif// VKEXEC_PRIVATE_BIND_RESOURCES_HPP
