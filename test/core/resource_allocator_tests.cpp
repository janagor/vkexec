#include <vkexec/buffer.hpp>
#include <vkexec/error.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec/tensor.hpp>

#include <catch2/catch_test_macros.hpp>
#include <stdexec/execution.hpp>
#include <vulkan/vulkan_core.h>

#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace resource_allocator_test {
// NOLINTBEGIN(misc-use-internal-linkage)

constexpr int k_wrong_value = 42;
constexpr float k_buffer_fill = 1.5F;
constexpr float k_tensor_fill = 2.0F;

// docs: custom allocator resource types begin
struct test_buffer
{
  std::vector<std::byte> bytes;

  [[nodiscard]] static auto handle() noexcept -> VkBuffer { return VK_NULL_HANDLE; }
  [[nodiscard]] auto size() const noexcept -> VkDeviceSize { return bytes.size(); }
  [[nodiscard]] auto mapped() noexcept -> std::span<std::byte> { return bytes; }
  [[nodiscard]] auto mapped() const noexcept -> std::span<std::byte const> { return bytes; }
  [[nodiscard]] static auto flush() -> vkexec::status { return {}; }
  [[nodiscard]] static auto invalidate() -> vkexec::status { return {}; }
  [[nodiscard]] static auto device_address() -> vkexec::result<VkDeviceAddress> { return VkDeviceAddress{}; }
};

struct test_image
{
  [[nodiscard]] static auto handle() noexcept -> VkImage { return VK_NULL_HANDLE; }
  [[nodiscard]] static auto format() noexcept -> VkFormat { return VK_FORMAT_R8G8B8A8_UNORM; }
};

struct test_allocator
{
  using buffer_type = test_buffer;
  using image_type = test_image;
  int buffer_allocations{};
  vkexec::buffer_create_info last_buffer_info{};
};
// docs: custom allocator resource types end

struct foreign_error_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures =
    stdexec::completion_signatures<stdexec::set_value_t(test_buffer), stdexec::set_error_t(std::string)>;
};

struct foreign_error_allocator
{
  using buffer_type = test_buffer;
};

[[nodiscard]] auto tag_invoke(vkexec::allocate_buffer_t /*tag*/,
  foreign_error_allocator & /*allocator*/,
  vkexec::buffer_create_info /*info*/) -> foreign_error_sender
{ return {}; }

struct infallible_allocator
{
  using buffer_type = test_buffer;
};

[[nodiscard]] auto
  tag_invoke(vkexec::allocate_buffer_t /*tag*/, infallible_allocator & /*allocator*/, vkexec::buffer_create_info info)
{ return stdexec::just(test_buffer{ std::vector<std::byte>(info.size) }); }

struct wrong_buffer_allocator
{
  using buffer_type = test_buffer;
};

[[nodiscard]] auto tag_invoke(vkexec::allocate_buffer_t /*tag*/,
  wrong_buffer_allocator & /*allocator*/,
  vkexec::buffer_create_info /*info*/)
{ return stdexec::just(k_wrong_value); }

struct wrong_image_allocator
{
  using image_type = test_image;
};

[[nodiscard]] auto tag_invoke(vkexec::allocate_image_t /*tag*/,
  wrong_image_allocator & /*allocator*/,
  vkexec::image_create_info const & /*info*/)
{ return stdexec::just(k_wrong_value); }

// docs: custom allocator customizations begin
[[nodiscard]] auto
  tag_invoke(vkexec::allocate_buffer_t /*tag*/, test_allocator &allocator, vkexec::buffer_create_info info)
{
  return vkexec::make_sender([&allocator, info]() -> vkexec::result<test_buffer> {
    ++allocator.buffer_allocations;
    allocator.last_buffer_info = info;
    return test_buffer{ std::vector<std::byte>(static_cast<std::size_t>(info.size)) };
  });
}

[[nodiscard]] auto
  tag_invoke(vkexec::allocate_image_t /*tag*/, test_allocator & /*allocator*/, vkexec::image_create_info const & /*info*/)
{
  return vkexec::make_sender([]() -> vkexec::result<test_image> { return test_image{}; });
}

static_assert(vkexec::resource_allocator<test_allocator>);
// docs: custom allocator customizations end
static_assert(vkexec::mapped_buffer_allocator<test_allocator>);
static_assert(std::same_as<decltype(std::declval<test_buffer const &>().mapped()), std::span<std::byte const>>);
static_assert(!vkexec::buffer_allocator<wrong_buffer_allocator>);
static_assert(!vkexec::image_allocator<wrong_image_allocator>);
static_assert(!vkexec::buffer_allocator<foreign_error_allocator>);
static_assert(vkexec::buffer_allocator<infallible_allocator>);
static_assert(
  stdexec::sender<decltype(vkexec::factory::make_buffer(std::declval<test_allocator &>(), 4, k_buffer_fill))>);
static_assert(
  stdexec::sender<decltype(vkexec::factory::make_tensor(std::declval<test_allocator &>(), 4, k_tensor_fill))>);

using buffer_sender = decltype(vkexec::factory::make_buffer(std::declval<test_allocator &>(), 4, k_buffer_fill));
using tensor_sender = decltype(vkexec::factory::make_tensor(std::declval<test_allocator &>(), 4, k_tensor_fill));

// NOLINTBEGIN(misc-include-cleaner)
static_assert(
  std::same_as<stdexec::error_types_of_t<buffer_sender, stdexec::env<>, std::tuple>, std::tuple<vkexec::error>>);
static_assert(
  std::same_as<stdexec::error_types_of_t<tensor_sender, stdexec::env<>, std::tuple>, std::tuple<vkexec::error>>);
// NOLINTEND(misc-include-cleaner)

// NOLINTEND(misc-use-internal-linkage)
}// namespace resource_allocator_test

TEST_CASE("generic resources use a non-VMA allocator without a context", "[vkexec][resource_allocator]")
{
  resource_allocator_test::test_allocator allocator;
  auto pending = vkexec::factory::make_buffer(allocator, 4, resource_allocator_test::k_buffer_fill);
  REQUIRE(allocator.buffer_allocations == 0);
  auto values = vkexec::try_sync_wait_value(pending);
  REQUIRE(values.has_value());
  REQUIRE(allocator.buffer_allocations == 1);
  REQUIRE(allocator.last_buffer_info.alignment == alignof(float));
  REQUIRE(values->size() == 4);
  REQUIRE(*values->data() == resource_allocator_test::k_buffer_fill);

  auto tensor =
    vkexec::try_sync_wait_value(vkexec::factory::make_tensor(allocator, 4, resource_allocator_test::k_tensor_fill));
  REQUIRE(tensor.has_value());
  REQUIRE(allocator.buffer_allocations == 3);
  REQUIRE(tensor->size() == 4);
  REQUIRE(*tensor->data() == resource_allocator_test::k_tensor_fill);

  auto invalid = vkexec::factory::make_tensor(allocator, std::numeric_limits<std::size_t>::max(), 1.0F);
  REQUIRE(allocator.buffer_allocations == 3);
  auto rejected = vkexec::try_sync_wait_value(invalid);
  REQUIRE_FALSE(rejected.has_value());
  REQUIRE(allocator.buffer_allocations == 3);
}
