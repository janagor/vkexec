#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_features/bundles/vulkan_13.hpp>
#include <vkexec_features/dynamic_rendering.hpp>
#include <vkexec_features/feature.hpp>
#include <vkexec_features/timeline_semaphore.hpp>

#include <memory>
#include <utility>

TEST_CASE("feat::configure enables timeline_semaphore on Vulkan 1.2+", "[vkexec][feature][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 2;
  vkexec::feat::configure<vkexec::feat::timeline_semaphore>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::context({ .requirements = std::move(requirements) }));
  REQUIRE(vkexec::feat::available<vkexec::feat::timeline_semaphore>(*ctx));
  REQUIRE(vkexec::feat::name<vkexec::feat::timeline_semaphore>() == "timeline_semaphore");
}

TEST_CASE("feat::configure enables dynamic_rendering on Vulkan 1.3+", "[vkexec][feature][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 3;
  vkexec::feat::configure<vkexec::feat::dynamic_rendering>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::context({ .requirements = std::move(requirements) }));
  REQUIRE(vkexec::feat::available<vkexec::feat::dynamic_rendering>(*ctx));
  REQUIRE(vkexec::feat::name<vkexec::feat::dynamic_rendering>() == "dynamic_rendering");
}

TEST_CASE("feat::configure_vulkan_13 enables registered 1.2 and 1.3 features", "[vkexec][feature][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 3;
  vkexec::feat::configure_vulkan_13(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::context({ .requirements = std::move(requirements) }));
  REQUIRE(vkexec::feat::available<vkexec::feat::timeline_semaphore>(*ctx));
  REQUIRE(vkexec::feat::available<vkexec::feat::buffer_device_address>(*ctx));
  REQUIRE(vkexec::feat::available<vkexec::feat::dynamic_rendering>(*ctx));
}
