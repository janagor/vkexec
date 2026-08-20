#ifndef VKEXEC_SUBMIT_HPP
#define VKEXEC_SUBMIT_HPP

namespace vkexec {

/// Pipe tag: turn a sync GPU sender into a completion sender that does not block `start()`.
struct submit_t
{
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr submit_t submit{};

}// namespace vkexec

#endif// VKEXEC_SUBMIT_HPP
