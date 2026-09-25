#ifndef VKEXEC_SUBMIT_HPP
#define VKEXEC_SUBMIT_HPP

namespace vkexec {

/**
 * Explicit submission adaptor for pass graphs.
 *
 * Pass graphs already submit asynchronously and complete on the context host
 * scheduler. This adaptor is retained for compatibility and produces the
 * explicit raw async form (`pass_graph_async_sender`), which completes on the
 * completion waiter without returning to the host scheduler.
 *
 * @see pass_graph_sender, pass_graph_async_sender
 */
struct submit_t
{
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr submit_t submit{};

}// namespace vkexec

#endif// VKEXEC_SUBMIT_HPP
