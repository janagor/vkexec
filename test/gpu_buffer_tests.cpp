#include <catch2/catch_test_macros.hpp>

#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/gpu_buffer.hpp>

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <exception>
#include <optional>
#include <string>

namespace {

constexpr VkDeviceSize k_bytes = 256;
constexpr std::byte k_marker{ static_cast<unsigned char>(0xAB) };

auto skip_if_no_vulkan(std::exception const &error) -> void
{ SKIP(std::string("Vulkan unavailable: ") + error.what()); }

}// namespace

TEST_CASE("gpu_buffer host_visible is mapped", "[vkexec][gpu_buffer][gpu]")
{
  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_no_vulkan(error); }

  auto buffer = vkexec::gpu_buffer::create(*ctx, k_bytes, vkexec::gpu_buffer_memory::host_visible);
  REQUIRE(buffer.handle() != VK_NULL_HANDLE);
  REQUIRE(buffer.size() == k_bytes);
  auto const mapped = buffer.mapped();
  REQUIRE(mapped.size() == static_cast<std::size_t>(k_bytes));
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
  mapped[0] = k_marker;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
  REQUIRE(mapped[0] == k_marker);
}

TEST_CASE("gpu_buffer device_local allocates without host mapping", "[vkexec][gpu_buffer][gpu]")
{
  std::optional<vkexec::context> ctx;
  VKEXEC_TRY { ctx.emplace(); }
  VKEXEC_CATCH(std::exception const &error) { skip_if_no_vulkan(error); }

  auto buffer = vkexec::gpu_buffer::create(*ctx, k_bytes, vkexec::gpu_buffer_memory::device_local);
  REQUIRE(buffer.handle() != VK_NULL_HANDLE);
  REQUIRE(buffer.size() == k_bytes);
  REQUIRE(buffer.mapped().empty());
}
