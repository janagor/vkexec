#include <catch2/catch_test_macros.hpp>

#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/submit_scope.hpp>

#include <vulkan/vulkan_core.h>

#include <array>
#include <span>

namespace {

constexpr VkDeviceSize k_byte_size = 64;

[[nodiscard]] auto fake_buffer(void *storage) -> VkBuffer { return static_cast<VkBuffer>(storage); }

}// namespace

TEST_CASE("storage_bindings_equal compares buffer bindings", "[vkexec][pass]")
{
  using vkexec::detail::storage_bindings_equal;

  char buffer_left{};
  char buffer_right{};
  vkexec::storage_binding const left_binding{
    .buffer = fake_buffer(&buffer_left), .byte_size = k_byte_size, .binding = 0
  };
  vkexec::storage_binding const right_binding{
    .buffer = fake_buffer(&buffer_right),
    .byte_size = k_byte_size,
    .binding = 0,
  };
  vkexec::storage_binding const left_copy = left_binding;

  REQUIRE(storage_bindings_equal(std::span{ &left_binding, 1 }, std::span{ &left_copy, 1 }));
  REQUIRE_FALSE(storage_bindings_equal(std::span{ &left_binding, 1 }, std::span{ &right_binding, 1 }));
  REQUIRE(
    storage_bindings_equal(std::span<vkexec::storage_binding const>{}, std::span<vkexec::storage_binding const>{}));

  vkexec::storage_binding const diff_binding{
    .buffer = left_binding.buffer,
    .byte_size = left_binding.byte_size,
    .binding = 1,
  };
  vkexec::storage_binding const diff_size_binding{
    .buffer = left_binding.buffer,
    .byte_size = left_binding.byte_size + 1,
    .binding = left_binding.binding,
  };

  REQUIRE_FALSE(storage_bindings_equal(std::span{ &left_binding, 1 }, std::span{ &diff_size_binding, 1 }));
  REQUIRE_FALSE(storage_bindings_equal(std::span{ &left_binding, 1 }, std::span{ &diff_binding, 1 }));

  std::array const bindings{ left_binding, right_binding };
  std::array const bindings_reordered{ right_binding, left_binding };
  REQUIRE_FALSE(storage_bindings_equal(std::span{ bindings }, std::span{ bindings_reordered }));
  REQUIRE(storage_bindings_equal(std::span{ bindings }, std::span{ bindings }));
}

TEST_CASE("write_storage_descriptors returns when no buffers are bound", "[vkexec][pass]")
{ vkexec::write_storage_descriptors(VK_NULL_HANDLE, VK_NULL_HANDLE, std::span<vkexec::storage_binding const>{}); }

TEST_CASE("pass batches group adjacent steps by resolved queue", "[vkexec][pass]")
{
  char graphics_storage{};
  char compute_storage{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  vkexec::queue_ref const graphics{ .queue = reinterpret_cast<VkQueue>(&graphics_storage), .family = 0 };
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  vkexec::queue_ref const compute{ .queue = reinterpret_cast<VkQueue>(&compute_storage), .family = 1 };
  vkexec::queue_affinity const unset{};
  vkexec::queue_affinity const graphics_affinity{ graphics };
  vkexec::queue_affinity const compute_affinity{ compute };

  auto const empty = vkexec::detail::plan_batches({}, graphics);
  REQUIRE(empty.batches.empty());

  std::array const one{ graphics_affinity };
  auto const one_plan = vkexec::detail::plan_batches(one, graphics);
  REQUIRE(one_plan.batches.size() == 1);
  REQUIRE(one_plan.batches.front().first_step == 0);
  REQUIRE(one_plan.batches.front().end_step == 1);

  std::array const repeated{ graphics_affinity, graphics_affinity, graphics_affinity };
  auto const repeated_plan = vkexec::detail::plan_batches(repeated, graphics);
  REQUIRE(repeated_plan.batches.size() == 1);
  REQUIRE(repeated_plan.batches.front().end_step == repeated.size());

  std::array const switch_once{ graphics_affinity, compute_affinity };
  auto const switch_plan = vkexec::detail::plan_batches(switch_once, graphics);
  REQUIRE(switch_plan.batches.size() == 2);
  REQUIRE(switch_plan.batches.front().end_step == 1);
  REQUIRE(switch_plan.batches.back().first_step == 1);

  std::array const switch_twice{
    graphics_affinity, graphics_affinity, compute_affinity, compute_affinity, graphics_affinity
  };
  auto const three_batches = vkexec::detail::plan_batches(switch_twice, graphics);
  REQUIRE(three_batches.batches.size() == 3);
  REQUIRE(three_batches.batches.front().end_step == 2);
  REQUIRE(three_batches.batches.at(1).first_step == 2);
  REQUIRE(three_batches.batches.at(1).end_step == 4);
  REQUIRE(three_batches.batches.back().first_step == 4);
  REQUIRE(three_batches.batches.back().end_step == switch_twice.size());

  std::array const defaults{ unset, unset };
  auto const default_plan = vkexec::detail::plan_batches(defaults, graphics);
  REQUIRE(default_plan.batches.size() == 1);
  REQUIRE(default_plan.batches.front().queue.queue == graphics.queue);
  REQUIRE(default_plan.batches.front().end_step == defaults.size());

  std::array const mixed{ unset, compute_affinity, compute_affinity, unset };
  auto const mixed_plan = vkexec::detail::plan_batches(mixed, graphics);
  REQUIRE(mixed_plan.batches.size() == 3);
  REQUIRE(mixed_plan.batches.front().queue.queue == graphics.queue);
  REQUIRE(mixed_plan.batches.at(1).queue.queue == compute.queue);
  REQUIRE(mixed_plan.batches.at(1).first_step == 1);
  REQUIRE(mixed_plan.batches.at(1).end_step == 3);
  REQUIRE(mixed_plan.batches.back().queue.queue == graphics.queue);
  REQUIRE(mixed_plan.batches.back().first_step == 3);

  auto planned = vkexec::detail::plan_execution(mixed.size(), mixed, graphics);
  REQUIRE(planned.has_value());
  REQUIRE(planned->batches.size() == mixed_plan.batches.size());
  REQUIRE_FALSE(vkexec::detail::plan_execution(mixed.size() - 1, mixed, graphics).has_value());
}
