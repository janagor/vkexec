#ifndef VKEXEC_GRAPHICS_SUBMIT_HPP
#define VKEXEC_GRAPHICS_SUBMIT_HPP

namespace vkexec {

/**
 * Selects asynchronous fence-based completion for graphics draw senders.
 *
 * Pass graphs are asynchronous by default and do not use this adaptor.
 */
struct submit_t
{
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr submit_t submit{};

}// namespace vkexec

#endif// VKEXEC_GRAPHICS_SUBMIT_HPP
