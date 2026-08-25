#include <catch2/catch_test_macros.hpp>

#include <vkexec/context.hpp>
#include <vkexec/error.hpp>
#include <vkexec/vulkan_requirements.hpp>

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

TEST_CASE("vkexec error category maps errc values", "[vkexec][error]")
{
  REQUIRE(std::string_view(vkexec::category().Name()) == "vkexec");
  REQUIRE(vkexec::MakeErrorCode(vkexec::errc::invalid_argument).Message() == "invalid argument");
  REQUIRE(vkexec::MakeErrorCode(vkexec::errc::io_error).Message() == "I/O error");
  REQUIRE(vkexec::MakeErrorCode(vkexec::errc::parse_error).Message() == "parse error");
  REQUIRE(vkexec::MakeErrorCode(vkexec::errc::unsupported).Message() == "unsupported operation");
  REQUIRE(vkexec::MakeErrorCode(vkexec::errc::out_of_range).Message() == "out of range");
  REQUIRE(vkexec::MakeErrorCode(vkexec::errc::empty_result).Message() == "empty result");
  REQUIRE(vkexec::MakeErrorCode(static_cast<vkexec::errc>(999)).Message() == "unknown vkexec error");
}

TEST_CASE("make_error prefers detail over category message", "[vkexec][error]")
{
  vkexec::error const with_detail = vkexec::make_error(vkexec::errc::parse_error, "bad glsl");
  REQUIRE(with_detail.message() == "bad glsl");

  vkexec::error const without_detail = vkexec::make_error(vkexec::errc::parse_error);
  REQUIRE(without_detail.message() == "parse error");
}

TEST_CASE("vulkan library floors document instance and device requests", "[vkexec][vulkan]")
{
  auto const headless_instance = vkexec::vulkan_library::required_headless_surface_instance_extensions();
  REQUIRE(headless_instance.size() == 2);
  REQUIRE(std::ranges::any_of(headless_instance, [](char const *name) -> bool {
    return std::strcmp(name, VK_KHR_SURFACE_EXTENSION_NAME) == 0;
  }));
  REQUIRE(std::ranges::any_of(headless_instance, [](char const *name) -> bool {
    return std::strcmp(name, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME) == 0;
  }));

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

TEST_CASE("context::try_create returns unsupported when requirements cannot be met", "[vkexec][error][gpu]")
{
  vkexec::vulkan_requirements requirements{};
  requirements.device_extensions = { "VK_VKEXEC_does_not_exist_EXT" };

  auto created = vkexec::context::try_create({ .requirements = std::move(requirements) });
  REQUIRE_FALSE(created.has_value());
  REQUIRE(created.error().code == vkexec::MakeErrorCode(vkexec::errc::unsupported));
  REQUIRE_FALSE(created.error().message().empty());
}

TEST_CASE("context::try_create succeeds for default requirements", "[vkexec][error][gpu]")
{
  auto created = vkexec::context::try_create();
  if (!created.has_value()) { SKIP(std::string("Vulkan unavailable: ") + std::string(created.error().message())); }
  REQUIRE((*created)->device() != VK_NULL_HANDLE);
}
