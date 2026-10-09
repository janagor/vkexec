#ifndef VKEXEC_PRESENTATION_ABORT_STATE_HPP
#define VKEXEC_PRESENTATION_ABORT_STATE_HPP

#include <cstdint>

namespace vkexec {

//! GPU state of an acquired image when a graph exits without presenting it.
enum class presentation_abort_state : std::uint8_t { acquired_only, final_submit_completed };

}// namespace vkexec

#endif// VKEXEC_PRESENTATION_ABORT_STATE_HPP
