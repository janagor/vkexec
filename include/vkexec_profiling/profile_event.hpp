#ifndef VKEXEC_PROFILING_PROFILE_EVENT_HPP
#define VKEXEC_PROFILING_PROFILE_EVENT_HPP

#include <cstdint>
#include <string>

namespace vkexec::profiling {

enum class sample_status { valid, unavailable, readback_error, invalid_timestamp };

struct profile_event
{
  std::string name;
  std::uint64_t submission_id{ 0 };
  std::uint32_t queue_family{ 0 };
  std::uint32_t region_index{ 0 };
  double duration_ns{ 0.0 };
  sample_status status{ sample_status::unavailable };
};

}// namespace vkexec::profiling

#endif// VKEXEC_PROFILING_PROFILE_EVENT_HPP
