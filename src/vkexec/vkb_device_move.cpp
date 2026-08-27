#include <VkBootstrap.h>

#include <vkexec/error.hpp>

namespace vkexec::detail {

// Out-of-line so clang CSA (per-TU) cannot see through LEAF's opaque result<> into
// vkb::Device's move assign — a known false positive with Boost.LEAF.
void move_from_leaf_device(vkb::Device &dest, leaf::result<vkb::Device> &src) { dest = std::move(*src); }

}// namespace vkexec::detail
