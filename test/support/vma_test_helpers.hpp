#ifndef VKEXEC_VMA_TEST_HELPERS_HPP
#define VKEXEC_VMA_TEST_HELPERS_HPP

#include "test_helpers.hpp"

#include <vkexec_vma/resources.hpp>

namespace vkexec::test {

[[nodiscard]] inline auto require_allocator(context &ctx) -> vma::allocator
{ return sync_wait_value(vma::factory::make_allocator(ctx)); }

}// namespace vkexec::test

#endif// VKEXEC_VMA_TEST_HELPERS_HPP
