#ifndef VKEXEC_TEST_PASS_DETAIL_TEST_SUPPORT_HPP
#define VKEXEC_TEST_PASS_DETAIL_TEST_SUPPORT_HPP

#include <catch2/catch_test_macros.hpp>

#include <vkexec/detail/resource_state_tracker.hpp>
#include <vkexec/pass.hpp>
// The lowering implementation depends on types defined by pass.hpp.
#include <vkexec/detail/pass_lowering.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/resource_use.hpp>
#include <vkexec/result.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit_scope.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

inline namespace vkexec_pass_detail_test_support {

constexpr VkDeviceSize k_byte_size = 64;
constexpr std::size_t k_first_step = 0;
constexpr std::size_t k_second_step = 1;
constexpr std::size_t k_pair_step_count = 2;
constexpr std::size_t k_single_step_count = 1;
constexpr std::size_t k_dag_step_count = 4;
constexpr std::size_t k_third_step = 2;
constexpr std::size_t k_fourth_step = 3;
constexpr std::uint32_t k_default_family = 0;
constexpr std::uint32_t k_other_family = 1;
constexpr std::uint32_t k_sampled_binding = 0;
constexpr std::uint32_t k_one_binding_count = 1;

[[nodiscard]] inline auto fake_buffer(void *storage) -> VkBuffer { return static_cast<VkBuffer>(storage); }

[[nodiscard]] inline auto fake_image(void *storage) -> VkImage { return static_cast<VkImage>(storage); }

constexpr VkImageSubresourceRange k_color_range{
  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
  .baseMipLevel = 0,
  .levelCount = 1,
  .baseArrayLayer = 0,
  .layerCount = 1,
};

}// namespace vkexec_pass_detail_test_support


#endif// VKEXEC_TEST_PASS_DETAIL_TEST_SUPPORT_HPP
