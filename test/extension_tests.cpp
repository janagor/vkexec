#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_extensions/descriptor_heap/extension.hpp>
#include <vkexec_extensions/extension.hpp>

#include <memory>
#include <utility>

TEST_CASE("ext::configure enables descriptor_heap when device supports it", "[vkexec][extension][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  vkexec::ext::configure<vkexec::ext::descriptor_heap>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::context::create({ .requirements = std::move(requirements) }));
  REQUIRE(vkexec::ext::available<vkexec::ext::descriptor_heap>(*ctx));
  REQUIRE(vkexec::ext::name<vkexec::ext::descriptor_heap>() == "descriptor_heap");
}
