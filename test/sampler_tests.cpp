#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/sampler.hpp>

#include <vulkan/vulkan_core.h>

#include <memory>

TEST_CASE("sampler creates a linear clamp sampler", "[vkexec][sampler][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto samp = vkexec::test::sync_wait_value(vkexec::sampler::create(*ctx));
  REQUIRE(samp.handle() != VK_NULL_HANDLE);
}
