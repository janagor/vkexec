#ifndef VKEXEC_PROFILING_CHROME_TRACE_HPP
#define VKEXEC_PROFILING_CHROME_TRACE_HPP

#include <vkexec/result.hpp>
#include <vkexec_profiling/profile_event.hpp>

#include <cstddef>
#include <iosfwd>
#include <span>

namespace vkexec::profiling {

struct trace_export_result
{
  std::size_t omitted_events{ 0 };
};

// Events must belong to one submission and queue family. Invalid samples are omitted.
[[nodiscard]] auto write_chrome_trace(std::ostream &output, std::span<profile_event const> events)
  -> result<trace_export_result>;

}// namespace vkexec::profiling

#endif// VKEXEC_PROFILING_CHROME_TRACE_HPP
