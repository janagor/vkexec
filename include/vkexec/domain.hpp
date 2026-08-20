#ifndef VKEXEC_DOMAIN_HPP
#define VKEXEC_DOMAIN_HPP

#include <stdexec/execution.hpp>

#include <utility>

namespace vkexec {

namespace ex = stdexec;

/// Scheduler domain: algorithms can rewrite senders to Vulkan-native forms (nvexec-style).
struct domain
{
  template<class OpTag, class Sender, class Env>
    requires requires(OpTag tag, Sender &&sndr, Env const &env) {
      lower_vkexec_sender(tag, static_cast<Sender &&>(sndr), env);
    }
  [[nodiscard]] static auto transform_sender(OpTag tag, Sender &&sndr, Env const &env)
    -> decltype(lower_vkexec_sender(tag, static_cast<Sender &&>(sndr), env))
  {
    return lower_vkexec_sender(tag, std::forward<Sender>(sndr), env);
  }

  template<class OpTag, class Sender, class Env>
    requires(!requires(OpTag tag, Sender &&sndr, Env const &env) {
      lower_vkexec_sender(tag, static_cast<Sender &&>(sndr), env);
    })
  [[nodiscard]] static auto transform_sender(OpTag tag, Sender &&sndr, Env const &env)
    -> decltype(ex::default_domain{}.transform_sender(tag, static_cast<Sender &&>(sndr), env))
  {
    return ex::default_domain{}.transform_sender(tag, std::forward<Sender>(sndr), env);
  }

  template<class Tag, class... Args>
  [[nodiscard]] static auto apply_sender(Tag tag, Args &&...args)
    -> decltype(ex::default_domain{}.apply_sender(tag, static_cast<Args &&>(args)...))
  {
    return ex::default_domain{}.apply_sender(tag, std::forward<Args>(args)...);
  }
};

}// namespace vkexec

#endif// VKEXEC_DOMAIN_HPP
