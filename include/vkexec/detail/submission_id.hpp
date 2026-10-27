#ifndef VKEXEC_DETAIL_SUBMISSION_ID_HPP
#define VKEXEC_DETAIL_SUBMISSION_ID_HPP

#include <cstdint>

namespace vkexec::detail {

//! Process-local identity shared by graph batches and manually submitted captures.
[[nodiscard]] auto allocate_submission_id() noexcept -> std::uint64_t;

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_SUBMISSION_ID_HPP
