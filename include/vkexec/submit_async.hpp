#ifndef VKEXEC_SUBMIT_ASYNC_HPP
#define VKEXEC_SUBMIT_ASYNC_HPP

namespace vkexec {

/// Pipe tag: turn a sync GPU sender into an async completion sender.
struct submit_async_t
{
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr submit_async_t submit_async{};

}// namespace vkexec

#endif// VKEXEC_SUBMIT_ASYNC_HPP
