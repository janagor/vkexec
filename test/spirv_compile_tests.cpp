#include <catch2/catch_test_macros.hpp>

#include <vkexec/spirv_compile.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace {

constexpr std::string_view k_trivial_compute = R"(
#version 450
layout(local_size_x = 1) in;
void main() {}
)";

[[nodiscard]] auto spirv_version(std::span<std::uint32_t const> spirv) -> std::uint32_t
{
  constexpr std::size_t k_spirv_version_word_index = 1;
  REQUIRE(!spirv.empty());
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access) -- version word index is fixed
  return spirv[k_spirv_version_word_index];
}

}// namespace

TEST_CASE("compile_glsl_to_spirv targets SPIR-V 1.0 for Vulkan 1.0", "[vkexec][spirv]")
{
  auto const spirv =
    vkexec::compile_glsl_to_spirv(k_trivial_compute, "trivial.comp", vkexec::shader_kind::compute, VK_API_VERSION_1_0);
  REQUIRE(spirv.has_value());
  REQUIRE(spirv_version(*spirv) == 0x00010000U);
}

TEST_CASE("compile_glsl_to_spirv targets SPIR-V 1.5 for Vulkan 1.2", "[vkexec][spirv]")
{
  auto const spirv =
    vkexec::compile_glsl_to_spirv(k_trivial_compute, "trivial.comp", vkexec::shader_kind::compute, VK_API_VERSION_1_2);
  REQUIRE(spirv.has_value());
  REQUIRE(spirv_version(*spirv) == 0x00010500U);
}

TEST_CASE("compile_glsl_to_spirv targets SPIR-V 1.6 for Vulkan 1.3", "[vkexec][spirv]")
{
  auto const spirv =
    vkexec::compile_glsl_to_spirv(k_trivial_compute, "trivial.comp", vkexec::shader_kind::compute, VK_API_VERSION_1_3);
  REQUIRE(spirv.has_value());
  REQUIRE(spirv_version(*spirv) == 0x00010600U);
}
