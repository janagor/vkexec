// NOLINTBEGIN(readability-function-cognitive-complexity)

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_extensions/descriptor_heap/extension.hpp>
#include <vkexec_extensions/extension.hpp>

#include <vulkan/vulkan_core.h>

#include <memory>
#include <string_view>
#include <utility>

TEST_CASE("descriptor heap configuration requests untyped shader support", "[vkexec][extension]")
{
  vkexec::vulkan_requirements requirements{};
  vkexec::ext::configure<vkexec::ext::descriptor_heap>(requirements);
  bool found{};
  for (char const *extension : requirements.optional_device_extensions) {
    if (std::string_view{ extension } == VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME) { found = true; }
  }
  REQUIRE(found);
}

TEST_CASE("ext::configure enables descriptor_heap when device supports it", "[vkexec][extension][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.api_version_major = 1;
  requirements.api_version_minor = 4;
  vkexec::ext::configure<vkexec::ext::descriptor_heap>(requirements);

  auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  // configure() requests the extension optionally; skip when the device lacks it.
  if (!vkexec::ext::available<vkexec::ext::descriptor_heap>(*ctx)) {
    ctx.reset();
    SKIP("descriptor_heap not supported on this device");
  }
  REQUIRE(vkexec::ext::name<vkexec::ext::descriptor_heap>() == "descriptor_heap");
}

TEST_CASE("extension proc tables follow context lifetimes", "[vkexec][extension][gpu]")
{
  struct fake_procs
  {
    int generation{};
  };

  constexpr int k_first_generation = 1;
  constexpr int k_context_count = 2;
  int loads{};
  for (int generation = k_first_generation; generation <= k_context_count; ++generation) {
    auto ctx = vkexec::test::sync_wait_value(vkexec::factory::make_context());
    auto load = [&loads, generation](VkDevice /*device*/) -> fake_procs {
      ++loads;
      return { .generation = generation };
    };
    auto const &first = ctx->cached_extension_procs<fake_procs>("test.fake_procs", load);
    auto const &again = ctx->cached_extension_procs<fake_procs>("test.fake_procs", load);
    REQUIRE(&first == &again);
    REQUIRE(first.generation == generation);
    REQUIRE(loads == generation);
  }
}

// NOLINTEND(readability-function-cognitive-complexity)
