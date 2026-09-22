#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

TEST_CASE("vkexec error category maps errc values", "[vkexec][error]")
{
  REQUIRE(std::string_view(vkexec::category().name()) == "vkexec");
  REQUIRE(vkexec::make_error_code(vkexec::errc::invalid_argument).message() == "invalid argument");
  REQUIRE(vkexec::make_error_code(vkexec::errc::io_error).message() == "I/O error");
  REQUIRE(vkexec::make_error_code(vkexec::errc::parse_error).message() == "parse error");
  REQUIRE(vkexec::make_error_code(vkexec::errc::unsupported).message() == "unsupported operation");
  REQUIRE(vkexec::make_error_code(vkexec::errc::out_of_range).message() == "out of range");
  REQUIRE(vkexec::make_error_code(vkexec::errc::empty_result).message() == "empty result");
  REQUIRE(boost::system::error_code(999, vkexec::category()).message() == "unknown vkexec error");
}

TEST_CASE("make_error prefers detail over category message", "[vkexec][error]")
{
  vkexec::error const with_detail = vkexec::make_error(vkexec::errc::parse_error, "bad glsl");
  REQUIRE(with_detail.message() == "bad glsl");

  vkexec::error const without_detail = vkexec::make_error(vkexec::errc::parse_error);
  REQUIRE(without_detail.message() == "parse error");
}

TEST_CASE("vulkan error category maps VkResult values", "[vkexec][error][vulkan]")
{
  REQUIRE(std::string_view(vkexec::vulkan_category().name()) == "vkexec.vulkan");
  REQUIRE(vkexec::make_vk_error_code(VK_SUCCESS).message() == "success");
  REQUIRE(vkexec::make_vk_error_code(VK_ERROR_DEVICE_LOST).message() == "device lost");
  REQUIRE(vkexec::make_vk_error_code(VK_ERROR_OUT_OF_DEVICE_MEMORY).message() == "out of device memory");
  REQUIRE(vkexec::make_vk_error_code(VK_ERROR_VALIDATION_FAILED_EXT).message() == "validation failed");
  REQUIRE(vkexec::make_vk_error_code(999).message() == "vulkan error");
}

TEST_CASE("make_vk_error attaches optional context detail", "[vkexec][error][vulkan]")
{
  vkexec::error const with_context = vkexec::make_vk_error(VK_ERROR_DEVICE_LOST, "submit failed");
  REQUIRE(with_context.code == vkexec::make_vk_error_code(VK_ERROR_DEVICE_LOST));
  REQUIRE(with_context.message() == "submit failed");

  vkexec::error const without_context = vkexec::make_vk_error(VK_ERROR_DEVICE_LOST, {});
  REQUIRE(without_context.message() == "device lost");
}

TEST_CASE("vulkan library floors document instance and device requests", "[vkexec][vulkan]")
{
  auto const headless_instance = vkexec::vulkan_library::required_headless_surface_instance_extensions();
  REQUIRE(headless_instance.size() == 2);
  REQUIRE(std::ranges::any_of(
    headless_instance, [](char const *name) -> bool { return std::strcmp(name, VK_KHR_SURFACE_EXTENSION_NAME) == 0; }));
  REQUIRE(std::ranges::any_of(headless_instance,
    [](char const *name) -> bool { return std::strcmp(name, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME) == 0; }));

  auto const present_device = vkexec::vulkan_library::required_presentation_device_extensions();
  REQUIRE(present_device.size() == 1);
  REQUIRE(std::strcmp(present_device.front(), VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0);

  REQUIRE(vkexec::vulkan_library::required_instance_extensions().empty());
  REQUIRE(vkexec::vulkan_library::required_device_extensions().empty());

  vkexec::vulkan_requirements const defaults{};
  REQUIRE(defaults.api_version_major == 1);
  REQUIRE(defaults.api_version_minor == 0);
  REQUIRE(defaults.instance_extensions.empty());
  REQUIRE(defaults.device_extensions.empty());
  REQUIRE(defaults.optional_device_extensions.empty());
  REQUIRE(defaults.required_extension_features.empty());
  REQUIRE(defaults.optional_extension_features.empty());
}

TEST_CASE("factory::make_context returns unsupported when requirements cannot be met", "[vkexec][error][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.device_extensions = { "VK_VKEXEC_does_not_exist_EXT" };

  auto outcome = vkexec::try_sync_wait(vkexec::factory::make_context({ .requirements = std::move(requirements) }));
  REQUIRE(outcome.failed());
  vkexec::error const err = outcome.take_error();
  REQUIRE(err.code == vkexec::make_error_code(vkexec::errc::unsupported));
  REQUIRE_FALSE(err.message().empty());
}

TEST_CASE("factory::make_context succeeds for default requirements", "[vkexec][error][gpu]")
{
  auto ctx = vkexec::test::require_context();
  REQUIRE(ctx->device() != VK_NULL_HANDLE);
}
