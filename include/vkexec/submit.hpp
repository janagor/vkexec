#ifndef VKEXEC_SUBMIT_HPP
#define VKEXEC_SUBMIT_HPP

namespace vkexec {

/**
 * Pipe tag that turns a sync GPU sender into a completion sender.
 *
 * Use as `| vkexec::submit` so `start()` does not block on the GPU; completion
 * is delivered asynchronously via the context fence agent.
 *
 * @see detail::submit_fence_sender
 */
struct submit_t
{
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr submit_t submit{};

}// namespace vkexec

#endif// VKEXEC_SUBMIT_HPP
