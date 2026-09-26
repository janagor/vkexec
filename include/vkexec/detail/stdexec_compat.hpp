#ifndef VKEXEC_DETAIL_STDEXEC_COMPAT_HPP
#define VKEXEC_DETAIL_STDEXEC_COMPAT_HPP

//! \file
//! Quarantined compatibility with the supported NVIDIA/stdexec revision.

#include <stdexec/execution.hpp>

namespace vkexec::detail::stdexec_compat {

// NVIDIA/stdexec implementation detail. Required to preserve the root
// environment semantics of sync_wait.
using root_t = stdexec::__root_t;

}// namespace vkexec::detail::stdexec_compat

#endif// VKEXEC_DETAIL_STDEXEC_COMPAT_HPP
