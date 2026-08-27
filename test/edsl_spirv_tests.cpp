#include <catch2/catch_test_macros.hpp>

#include <vkexec_edsl/spirv.hpp>

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view k_trivial_compute = R"(#version 450
layout(local_size_x = 1) in;
void main() {}
)";

/// SPIR-V magic is word0; version is word1 as 0x00MMmm00.
constexpr std::uint32_t k_spirv_version_major_shift = 16U;
constexpr std::uint32_t k_spirv_version_minor_shift = 8U;

auto spirv_version_word(std::vector<std::uint32_t> const &words) -> std::uint32_t
{
  REQUIRE(words.size() >= 2);
  return words.at(1);
}

auto make_spirv_version(std::uint32_t major, std::uint32_t minor) -> std::uint32_t
{ return (major << k_spirv_version_major_shift) | (minor << k_spirv_version_minor_shift); }

}// namespace

TEST_CASE("compile_glsl_to_spirv targets SPIR-V 1.0 for Vulkan 1.0", "[vkexec][edsl][spirv]")
{
  auto const spirv = vkexec::edsl::compile_glsl_to_spirv(
    k_trivial_compute, "trivial.comp", vkexec::edsl::shader_kind::compute, VK_API_VERSION_1_0);
  REQUIRE(spirv.has_value());
  REQUIRE(spirv_version_word(*spirv) == make_spirv_version(1, 0));
}

TEST_CASE("compile_glsl_to_spirv targets SPIR-V 1.5 for Vulkan 1.2", "[vkexec][edsl][spirv]")
{
  auto const spirv = vkexec::edsl::compile_glsl_to_spirv(
    k_trivial_compute, "trivial.comp", vkexec::edsl::shader_kind::compute, VK_API_VERSION_1_2);
  REQUIRE(spirv.has_value());
  REQUIRE(spirv_version_word(*spirv) == make_spirv_version(1, 5));
}

TEST_CASE("compile_glsl_to_spirv targets SPIR-V 1.6 for Vulkan 1.3", "[vkexec][edsl][spirv]")
{
  auto const spirv = vkexec::edsl::compile_glsl_to_spirv(
    k_trivial_compute, "trivial.comp", vkexec::edsl::shader_kind::compute, VK_API_VERSION_1_3);
  REQUIRE(spirv.has_value());
  REQUIRE(spirv_version_word(*spirv) == make_spirv_version(1, 6));
}
