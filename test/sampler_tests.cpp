#include <catch2/catch_test_macros.hpp>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/sampler.hpp>

#include <vulkan/vulkan_core.h>

#include <exception>
#include <optional>
#include <string>

namespace {

auto skip_if_no_vulkan(std::exception const &error) -> void
{ SKIP(std::string("Vulkan unavailable: ") + error.what()); }

}// namespace

TEST_CASE("sampler creates a linear clamp sampler", "[vkexec][sampler][gpu]")
{
  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_no_vulkan(error); }

  auto samp = vkexec::sampler::create(*ctx);
  REQUIRE(samp.handle() != VK_NULL_HANDLE);
}
