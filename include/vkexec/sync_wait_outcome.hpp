#ifndef VKEXEC_SYNC_WAIT_OUTCOME_HPP
#define VKEXEC_SYNC_WAIT_OUTCOME_HPP

//! \file
//! Public alias for the non-throwing `sync_wait` / `try_sync_wait` outcome type.

#include <vkexec/detail/sync_wait_outcome.hpp>

namespace vkexec {

/**
 * Result of a non-throwing blocking wait: values, error, or stopped.
 *
 * @see try_sync_wait, sync_wait
 */
template<class... Values> using sync_wait_outcome = detail::sync_wait_outcome<Values...>;

}// namespace vkexec

#endif// VKEXEC_SYNC_WAIT_OUTCOME_HPP
