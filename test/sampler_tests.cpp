#include <catch2/catch_test_macros.hpp>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/sampler.hpp>

#include <vulkan/vulkan_core.h>

#include <memory>
#include <string>

namespace {

auto skip_if_no_vulkan(vkexec::error const &err) -> void
{ SKIP(std::string("Vulkan unavailable: ") + std::string(err.message())); }

}// namespace

TEST_CASE("sampler creates a linear clamp sampler", "[vkexec][sampler][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto samp_result = vkexec::sampler::create(ctx);
  REQUIRE(samp_result.has_value());
  REQUIRE(samp_result->handle() != VK_NULL_HANDLE);
}
