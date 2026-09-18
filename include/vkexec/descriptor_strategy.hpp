#ifndef VKEXEC_DESCRIPTOR_STRATEGY_HPP
#define VKEXEC_DESCRIPTOR_STRATEGY_HPP

//! \file
//! Core descriptor-set strategy selector.

namespace vkexec {

//! Selects the core `VkDescriptorSet` descriptor backend.
struct descriptor_sets_t
{
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr descriptor_sets_t descriptor_sets{};

}// namespace vkexec

#endif// VKEXEC_DESCRIPTOR_STRATEGY_HPP
