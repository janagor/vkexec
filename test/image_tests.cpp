#include <catch2/catch_test_macros.hpp>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/image.hpp>
#include <vkexec/image_view.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>
#include <string>

namespace {

constexpr std::uint32_t k_width = 64;
constexpr std::uint32_t k_height = 48;

auto skip_if_no_vulkan(vkexec::error const &err) -> void
{ SKIP(std::string("Vulkan unavailable: ") + std::string(err.message())); }

}// namespace

TEST_CASE("image color_storage allocates a device-local target", "[vkexec][image][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto img_result = vkexec::image::create(ctx,
    vkexec::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::image_usage::color_storage,
    });
  REQUIRE(img_result.has_value());
  REQUIRE(img_result->handle() != VK_NULL_HANDLE);
  REQUIRE(img_result->extent().width == k_width);
  REQUIRE(img_result->extent().height == k_height);
  REQUIRE(img_result->format() == VK_FORMAT_R16G16B16A16_SFLOAT);
}

TEST_CASE("image depth allocates a depth attachment", "[vkexec][image][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto img_result = vkexec::image::create(ctx,
    vkexec::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::image_usage::depth,
    });
  REQUIRE(img_result.has_value());
  REQUIRE(img_result->handle() != VK_NULL_HANDLE);
  REQUIRE(img_result->format() == VK_FORMAT_D32_SFLOAT);
}

TEST_CASE("image_view wraps a color image", "[vkexec][image][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto img_result = vkexec::image::create(ctx,
    vkexec::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::image_usage::color_storage,
    });
  REQUIRE(img_result.has_value());
  auto view_result = vkexec::image_view::create(ctx, *img_result);
  REQUIRE(view_result.has_value());
  REQUIRE(view_result->handle() != VK_NULL_HANDLE);
}
