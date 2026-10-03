#include "test_helpers.hpp"
#include "vma_test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_features/buffer_device_address.hpp>
#include <vkexec_features/feature.hpp>
#include <vkexec_vma/gpu_buffer.hpp>

#include <vulkan/vulkan_core.h>

#include <utility>

namespace {

constexpr VkDeviceSize k_bytes = 256;

}// namespace

TEST_CASE("gpu_buffer device_address works with bufferDeviceAddress enabled", "[vkexec][gpu_buffer][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 2;
  vkexec::feat::configure<vkexec::feat::buffer_device_address>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto buffer = vkexec::test::sync_wait_value(vkexec::vma::factory::make_gpu_buffer(allocator,
    vkexec::vma::gpu_buffer_create_info{
      .size = k_bytes,
      .memory = vkexec::vma::buffer_memory::device_local,
      .shader_device_address = true,
    }));
  auto const addr_result = buffer.device_address();
  REQUIRE(addr_result.has_value());
  REQUIRE(*addr_result != 0);
}
