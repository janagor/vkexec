#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/image.hpp>
#include <vkexec/image_view.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>

namespace {

constexpr std::uint32_t k_width = 64;
constexpr std::uint32_t k_height = 48;

}// namespace

TEST_CASE("image color_storage allocates a device-local target", "[vkexec][image][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto img = vkexec::test::sync_wait_value(vkexec::factory::image(*ctx,
    vkexec::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::image_usage::color_storage,
    }));
  REQUIRE(img.handle() != VK_NULL_HANDLE);
  REQUIRE(img.extent().width == k_width);
  REQUIRE(img.extent().height == k_height);
  REQUIRE(img.format() == VK_FORMAT_R16G16B16A16_SFLOAT);
}

TEST_CASE("image depth allocates a depth attachment", "[vkexec][image][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto img = vkexec::test::sync_wait_value(vkexec::factory::image(*ctx,
    vkexec::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::image_usage::depth,
    }));
  REQUIRE(img.handle() != VK_NULL_HANDLE);
  REQUIRE(img.format() == VK_FORMAT_D32_SFLOAT);
}

TEST_CASE("image_view wraps a color image", "[vkexec][image][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto img = vkexec::test::sync_wait_value(vkexec::factory::image(*ctx,
    vkexec::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::image_usage::color_storage,
    }));
  auto view = vkexec::test::sync_wait_value(vkexec::factory::image_view(*ctx, img));
  REQUIRE(view.handle() != VK_NULL_HANDLE);
}
