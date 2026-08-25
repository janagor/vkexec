#include <catch2/catch_test_macros.hpp>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/image.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <exception>
#include <optional>
#include <string>

namespace {

constexpr std::uint32_t k_width = 64;
constexpr std::uint32_t k_height = 48;

auto skip_if_no_vulkan(std::exception const &error) -> void
{ SKIP(std::string("Vulkan unavailable: ") + error.what()); }

}// namespace

TEST_CASE("image color_storage allocates a device-local target", "[vkexec][image][gpu]")
{
  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_no_vulkan(error); }

  auto img = vkexec::image::create(*ctx,
    vkexec::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::image_usage::color_storage,
    });
  REQUIRE(img.handle() != VK_NULL_HANDLE);
  REQUIRE(img.extent().width == k_width);
  REQUIRE(img.extent().height == k_height);
  REQUIRE(img.format() == VK_FORMAT_R16G16B16A16_SFLOAT);
}

TEST_CASE("image depth allocates a depth attachment", "[vkexec][image][gpu]")
{
  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_no_vulkan(error); }

  auto img = vkexec::image::create(*ctx,
    vkexec::image_create_info{
      .width = k_width,
      .height = k_height,
      .usage = vkexec::image_usage::depth,
    });
  REQUIRE(img.handle() != VK_NULL_HANDLE);
  REQUIRE(img.format() == VK_FORMAT_D32_SFLOAT);
}
