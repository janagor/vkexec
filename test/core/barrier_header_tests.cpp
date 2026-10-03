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
static_assert(std::is_same_v<decltype(vkexec::buffer_barrier_params{}.size), VkDeviceSize>);
static_assert(vkexec::buffer_barrier_params{}.size == VK_WHOLE_SIZE);
static_assert(vkexec::buffer_barrier_params{}.src_queue_family == VK_QUEUE_FAMILY_IGNORED);
static_assert(vkexec::image_barrier_params{}.range.levelCount == 1);
static_assert(vkexec::image_barrier_params{}.range.layerCount == 1);
static_assert(vkexec::image_barrier_params{}.dst_queue_family == VK_QUEUE_FAMILY_IGNORED);
static_assert(requires { vkexec::barrier::buffer(vkexec::buffer_barrier_params{}); });
static_assert(requires { vkexec::barrier::image(vkexec::image_barrier_params{}); });
static_assert(std::is_same_v<decltype(vkexec::barrier::compute_to_compute(std::declval<vkexec::context &>(),
                               std::declval<VkCommandBuffer>())),
  vkexec::status>);
