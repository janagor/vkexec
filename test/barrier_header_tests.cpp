#include <vkexec/barrier.hpp>
#include <vkexec/context.hpp>
#include <vkexec/result.hpp>

#include <vulkan/vulkan_core.h>

#include <type_traits>
#include <utility>

static_assert(requires { vkexec::barrier::compute_to_compute(); });

static_assert(std::is_same_v<decltype(vkexec::memory_barrier_params{}.src_stage), VkPipelineStageFlags2>);
static_assert(std::is_same_v<decltype(vkexec::memory_barrier_params{}.dst_access), VkAccessFlags2>);
static_assert(std::is_same_v<decltype(vkexec::image_barrier_params{}.src_stage), VkPipelineStageFlags2>);
static_assert(std::is_same_v<decltype(vkexec::barrier::compute_to_compute(std::declval<vkexec::context &>(),
                               std::declval<VkCommandBuffer>())),
  vkexec::status>);
