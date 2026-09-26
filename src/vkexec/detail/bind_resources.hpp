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
  handles::compute_pipeline const *pipe{ nullptr };
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

}// namespace vkexec::detail

#endif// VKEXEC_PRIVATE_BIND_RESOURCES_HPP
