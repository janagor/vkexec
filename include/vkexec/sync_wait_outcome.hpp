#ifndef VKEXEC_SYNC_WAIT_OUTCOME_HPP
#define VKEXEC_SYNC_WAIT_OUTCOME_HPP

#include <vkexec/detail/sync_wait_outcome.hpp>

namespace vkexec {

template<class... Values> using sync_wait_outcome = detail::sync_wait_outcome<Values...>;

}// namespace vkexec

#endif// VKEXEC_SYNC_WAIT_OUTCOME_HPP
