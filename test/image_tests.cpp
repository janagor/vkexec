#include "test_helpers.hpp"
#include "vma_test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/image_view.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec_vma/image.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>

namespace {

constexpr std::uint32_t k_width = 64;
constexpr std::uint32_t k_height = 48;
constexpr std::uint32_t k_cube_layers = 6;
constexpr std::uint32_t k_extra_cube_layer_count = 7;
constexpr std::uint32_t k_small_extent = 16;

}// namespace

TEST_CASE("image color_storage allocates a device-local target", "[vkexec][image][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto img = vkexec::test::sync_wait_value(vkexec::vma::factory::make_image(allocator,
    vkexec::vma::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::vma::image_usage::color_storage,
    }));
  REQUIRE(img.handle() != VK_NULL_HANDLE);
  REQUIRE(img.extent().width == k_width);
  REQUIRE(img.extent().height == k_height);
  REQUIRE(img.format() == VK_FORMAT_R16G16B16A16_SFLOAT);
}

TEST_CASE("image depth allocates a depth attachment", "[vkexec][image][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto img = vkexec::test::sync_wait_value(vkexec::vma::factory::make_image(allocator,
    vkexec::vma::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::vma::image_usage::depth,
    }));
  REQUIRE(img.handle() != VK_NULL_HANDLE);
  REQUIRE(img.format() == VK_FORMAT_D32_SFLOAT);
}

TEST_CASE("generic image retains mip and layer metadata", "[vkexec][image][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto img = vkexec::test::sync_wait_value(vkexec::allocate_image(allocator,
    vkexec::image_create_info{
      .extent = { .width = k_width, .height = k_width, .depth = 1 },
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .mip_levels = 4,
      .array_layers = k_cube_layers,
      .flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
    }));
  REQUIRE(img.extent_3d().width == k_width);
  REQUIRE(img.mip_levels() == 4);
  REQUIRE(img.array_layers() == k_cube_layers);
  REQUIRE(img.type() == VK_IMAGE_TYPE_2D);
  REQUIRE(img.samples() == VK_SAMPLE_COUNT_1_BIT);
  REQUIRE(img.flags() == VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
  REQUIRE(img.usage_flags() == (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
  auto view = vkexec::test::sync_wait_value(vkexec::vma::factory::make_image_view(*ctx,
    img,
    vkexec::image_view_create_info{
      .type = VK_IMAGE_VIEW_TYPE_CUBE,
      .range = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 4,
        .baseArrayLayer = 0,
        .layerCount = k_cube_layers },
    }));
  REQUIRE(view.handle() != VK_NULL_HANDLE);
  auto automatic_view = vkexec::test::sync_wait_value(vkexec::vma::factory::make_image_view(*ctx, img));
  REQUIRE(automatic_view.handle() != VK_NULL_HANDLE);
}

TEST_CASE("image_view wraps a color image", "[vkexec][image][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto img = vkexec::test::sync_wait_value(vkexec::vma::factory::make_image(allocator,
    vkexec::vma::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::vma::image_usage::color_storage,
    }));
  auto view = vkexec::test::sync_wait_value(vkexec::vma::factory::make_image_view(*ctx, img));
  REQUIRE(view.handle() != VK_NULL_HANDLE);
}

TEST_CASE("automatic view supports cube-compatible images with seven layers", "[vkexec][image][gpu]")
{
  auto ctx = vkexec::test::require_context();
  auto allocator = vkexec::test::require_allocator(*ctx);
  auto img = vkexec::test::sync_wait_value(vkexec::allocate_image(allocator,
    vkexec::image_create_info{
      .extent = { .width = k_small_extent, .height = k_small_extent, .depth = 1 },
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
      .array_layers = k_extra_cube_layer_count,
      .flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
    }));
  auto view = vkexec::test::sync_wait_value(vkexec::vma::factory::make_image_view(*ctx, img));
  REQUIRE(view.handle() != VK_NULL_HANDLE);
}
