#ifndef VKEXEC_DETAIL_CAPABILITY_ID_HPP
#define VKEXEC_DETAIL_CAPABILITY_ID_HPP

#include <cstdint>

namespace vkexec::detail {

enum class capability_id : std::uint8_t {
  none,
  synchronization2,
  timeline_semaphore,
  buffer_device_address,
  dynamic_rendering,
};

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_CAPABILITY_ID_HPP
