#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_features/feature.hpp>
#include <vkexec_features/timeline_semaphore.hpp>

#include <utility>

TEST_CASE("owned context reports enabled capabilities", "[vkexec][capabilities][gpu]")
{
  auto ctx = vkexec::test::require_context();
  REQUIRE(ctx->capabilities().api_version == ctx->api_version());
  REQUIRE_FALSE(ctx->capabilities().timeline_semaphore);
  REQUIRE_FALSE(vkexec::feat::available<vkexec::feat::timeline_semaphore>(*ctx));
}

TEST_CASE("adopted context uses declared capabilities", "[vkexec][capabilities][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  vkexec::feat::configure<vkexec::feat::timeline_semaphore>(requirements);
  auto owner =
    vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  REQUIRE(owner->capabilities().timeline_semaphore);
  vkexec::context_adopt_info info{};
  info.instance = owner->instance();
  info.physical_device = owner->physical_device();
  info.device = owner->device();
  info.capabilities = owner->capabilities();
  info.compute_queue = owner->compute_queue();
  info.compute_queue_family = owner->queue_family();
  info.graphics_queue = owner->graphics_queue();
  info.graphics_queue_family = owner->graphics_queue_family();
  info.present_queue = owner->present_queue();
  info.present_queue_family = owner->present_queue_family();

  auto adopted = vkexec::test::sync_wait_value(vkexec::factory::adopt_context(info));
  REQUIRE(adopted->capabilities().api_version == owner->capabilities().api_version);
  REQUIRE(adopted->capabilities().synchronization2 == owner->capabilities().synchronization2);
  REQUIRE(adopted->capabilities().timeline_semaphore == owner->capabilities().timeline_semaphore);
  REQUIRE(vkexec::feat::available<vkexec::feat::timeline_semaphore>(*adopted));
  REQUIRE(adopted->capabilities().buffer_device_address == owner->capabilities().buffer_device_address);
  REQUIRE(adopted->capabilities().dynamic_rendering == owner->capabilities().dynamic_rendering);
  REQUIRE_FALSE(adopted->owns_device());

  adopted.reset();
  info.capabilities.timeline_semaphore = false;
  auto limited = vkexec::test::sync_wait_value(vkexec::factory::adopt_context(info));
  REQUIRE_FALSE(limited->capabilities().timeline_semaphore);
  REQUIRE_FALSE(vkexec::feat::available<vkexec::feat::timeline_semaphore>(*limited));
}
