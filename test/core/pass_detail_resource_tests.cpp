// NOLINTBEGIN(readability-function-cognitive-complexity)

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexec/execution.hpp>
#include <tuple>
#include <utility>
#include <vector>
#include <vulkan/vulkan_core.h>

#include "pass_detail_test_support.hpp"
#include "vkexec/barrier_params.hpp"
#include "vkexec/detail/pass_execution.hpp"
#include "vkexec/detail/pass_lowering.hpp"
#include "vkexec/detail/pass_steps.hpp"
#include "vkexec/detail/pass_types.hpp"
#include "vkexec/detail/resource_state_tracker.hpp"
#include "vkexec/detail/resource_state_tracker_helpers.hpp"
#include "vkexec/detail/sender_expr.hpp"
#include "vkexec/pass.hpp"
#include "vkexec/pipeline.hpp"
#include "vkexec/queue_submit.hpp"
#include "vkexec/resource_table.hpp"
#include "vkexec/resource_use.hpp"
#include "vkexec/result.hpp"
#include "vkexec/scheduler.hpp"
#include "vkexec/submit_scope.hpp"

namespace vkexec {
class context;
}// namespace vkexec

namespace {

auto add_resource_dependencies_reference(vkexec::detail::execution_plan &plan,
  vkexec::detail::resource_sync_plan const &sync) -> void
{
  for (auto const &[source, destination] : sync.dependencies) {
    std::size_t source_batch{};
    std::size_t destination_batch{};
    for (std::size_t index{}; index < plan.batches.size(); ++index) {
      auto const &batch = plan.batches.at(index);
      if (source >= batch.first_step && source < batch.end_step) { source_batch = index; }
      if (destination >= batch.first_step && destination < batch.end_step) { destination_batch = index; }
    }
    if (source_batch == destination_batch) { continue; }
    auto &incoming = plan.batches.at(destination_batch).predecessors;
    if (std::ranges::find(incoming, source_batch) == incoming.end()) { incoming.push_back(source_batch); }
  }
}

auto require_same_batches(vkexec::detail::execution_plan const &actual, vkexec::detail::execution_plan const &expected)
  -> void
{
  REQUIRE(actual.batches.size() == expected.batches.size());
  for (std::size_t index{}; index < actual.batches.size(); ++index) {
    auto const &lhs = actual.batches.at(index);
    auto const &rhs = expected.batches.at(index);
    REQUIRE(lhs.queue.queue == rhs.queue.queue);
    REQUIRE(lhs.queue.family == rhs.queue.family);
    REQUIRE(lhs.first_step == rhs.first_step);
    REQUIRE(lhs.end_step == rhs.end_step);
    REQUIRE(lhs.predecessors == rhs.predecessors);
  }
}

auto require_reference_match(vkexec::detail::execution_plan const &initial,
  vkexec::detail::resource_sync_plan const &sync) -> void
{
  auto expected = initial;
  auto actual = initial;
  auto const original_dependencies = sync.dependencies;
  auto const original_step_count = sync.steps.size();
  add_resource_dependencies_reference(expected, sync);
  vkexec::detail::add_resource_dependencies(actual, sync);
  require_same_batches(actual, expected);
  REQUIRE(sync.dependencies == original_dependencies);
  REQUIRE(sync.steps.size() == original_step_count);
}

}// namespace

TEST_CASE("resource dependency mapping handles empty and unchanged plans", "[vkexec][pass]")
{
  vkexec::detail::execution_plan empty{};
  vkexec::detail::resource_sync_plan sync{};
  require_reference_match(empty, sync);

  sync.steps.resize(k_single_step_count);
  empty.batches.push_back(
    { .queue = {}, .first_step = k_first_step, .end_step = k_single_step_count, .predecessors = {} });
  require_reference_match(empty, sync);

  sync.steps.resize(k_dag_step_count);
  empty.batches.push_back(
    { .queue = {}, .first_step = k_single_step_count, .end_step = k_pair_step_count, .predecessors = {} });
  empty.batches.push_back(
    { .queue = {}, .first_step = k_pair_step_count, .end_step = k_dag_step_count, .predecessors = {} });
  require_reference_match(empty, sync);
}

TEST_CASE("resource dependency mapping preserves multi-step batch order and duplicates", "[vkexec][pass]")
{
  constexpr std::size_t k_first_end = 3;
  constexpr std::size_t k_second_end = 5;
  constexpr std::size_t k_step_count = 8;
  constexpr std::size_t k_last_step = 7;
  constexpr std::size_t k_middle_step = 4;
  constexpr std::size_t k_later_step = 6;
  vkexec::detail::execution_plan plan{};
  plan.batches = { { .queue = { .family = k_default_family },
                     .first_step = k_first_step,
                     .end_step = k_first_end,
                     .predecessors = {} },
    { .queue = { .family = k_other_family }, .first_step = k_first_end, .end_step = k_second_end, .predecessors = {} },
    { .queue = { .family = k_default_family },
      .first_step = k_second_end,
      .end_step = k_step_count,
      .predecessors = {} } };
  vkexec::detail::resource_sync_plan sync{};
  sync.steps.resize(k_step_count);
  sync.dependencies = { { k_second_step, k_third_step },
    { k_second_step, k_middle_step },
    { k_first_end, k_later_step },
    { k_first_step, k_last_step },
    { k_second_step, k_middle_step } };
  require_reference_match(plan, sync);
  vkexec::detail::add_resource_dependencies(plan, sync);
  REQUIRE(plan.batches.at(k_first_step).predecessors.empty());
  REQUIRE(plan.batches.at(k_second_step).predecessors == std::vector<std::size_t>{ k_first_step });
  REQUIRE(plan.batches.at(k_third_step).predecessors == std::vector<std::size_t>{ k_second_step, k_first_step });
  require_reference_match(plan, sync);

  plan.batches.at(k_second_step).predecessors = { k_first_step };
  plan.batches.at(k_third_step).predecessors = { k_first_step };
  require_reference_match(plan, sync);
  vkexec::detail::add_resource_dependencies(plan, sync);
  REQUIRE(plan.batches.at(k_third_step).predecessors == std::vector<std::size_t>{ k_first_step, k_second_step });
}

TEST_CASE("resource dependency mapping preserves fan-in", "[vkexec][pass]")
{
  vkexec::detail::execution_plan plan{};
  vkexec::detail::resource_sync_plan sync{};
  sync.steps.resize(k_dag_step_count);
  for (std::size_t step{}; step < k_dag_step_count; ++step) {
    plan.batches.push_back({ .queue = { .family = step % k_pair_step_count == 0 ? k_default_family : k_other_family },
      .first_step = step,
      .end_step = step + k_single_step_count,
      .predecessors = {} });
  }
  sync.dependencies = {
    { k_first_step, k_fourth_step }, { k_second_step, k_fourth_step }, { k_third_step, k_fourth_step }
  };
  require_reference_match(plan, sync);
  vkexec::detail::add_resource_dependencies(plan, sync);
  REQUIRE(plan.batches.at(k_fourth_step).predecessors
          == std::vector<std::size_t>{ k_first_step, k_second_step, k_third_step });
}

TEST_CASE("resource dependency mapping matches scan reference on valid partitions", "[vkexec][pass]")
{
  constexpr std::uint32_t k_seed = 0xC0FFEE;
  constexpr std::uint32_t k_multiplier = 1664525;
  constexpr std::uint32_t k_increment = 1013904223;
  constexpr std::size_t k_trials_per_layout = 32;
  constexpr std::array<std::size_t, 6> k_step_counts{ 1, 2, 8, 32, 128, 512 };
  std::uint32_t state = k_seed;
  auto next = [&]() -> std::uint32_t {
    state = (state * k_multiplier) + k_increment;
    return state;
  };
  for (auto const step_count : k_step_counts) {
    for (std::size_t trial{}; trial < k_trials_per_layout; ++trial) {
      vkexec::detail::execution_plan plan{};
      vkexec::detail::resource_sync_plan sync{};
      sync.steps.resize(step_count);
      std::size_t first{};
      while (first < step_count) {
        auto const remaining = step_count - first;
        auto const length = trial == 0 ? k_single_step_count : k_single_step_count + (next() % remaining);
        auto const index = plan.batches.size();
        std::vector<std::size_t> predecessors;
        for (std::size_t previous{}; previous < index; ++previous) {
          if (next() % k_dag_step_count == 0) { predecessors.push_back(previous); }
        }
        plan.batches.push_back(
          { .queue = { .family = next() % k_pair_step_count == 0 ? k_default_family : k_other_family },
            .first_step = first,
            .end_step = first + length,
            .predecessors = std::move(predecessors) });
        first += length;
      }
      REQUIRE(vkexec::detail::validate_execution_batches(plan, step_count));
      if (trial == 0) {
        for (std::size_t step = k_single_step_count; step < step_count; ++step) {
          sync.dependencies.emplace_back(step - k_single_step_count, step);
        }
      }
      auto const edge_count = trial * step_count / k_dag_step_count;
      for (std::size_t edge{}; edge < edge_count; ++edge) {
        auto const source = static_cast<std::size_t>(next()) % step_count;
        auto const destination = source + (static_cast<std::size_t>(next()) % (step_count - source));
        if (source == destination) { continue; }
        sync.dependencies.emplace_back(source, destination);
        if (edge % k_dag_step_count == 0) { sync.dependencies.emplace_back(source, destination); }
      }
      require_reference_match(plan, sync);
    }
  }
}

TEST_CASE("ownership transfer collapses a reader frontier to one release", "[vkexec][resource-state][pass]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::detail::resource_state_tracker tracker{ k_dag_step_count };
  REQUIRE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::storage_compute), k_first_step, k_default_family));
  REQUIRE(tracker.use(vkexec::read(buffer, vkexec::buffer_usage::uniform_compute), k_second_step, k_default_family));
  REQUIRE(tracker.use(vkexec::read(buffer, vkexec::buffer_usage::uniform_fragment), k_third_step, k_default_family));
  REQUIRE(tracker.use(vkexec::read(buffer, vkexec::buffer_usage::uniform_compute), k_fourth_step, k_other_family));

  auto const plan = std::move(tracker).finish();
  REQUIRE(std::ranges::find(plan.dependencies, std::pair{ k_second_step, k_third_step }) != plan.dependencies.end());
  REQUIRE(plan.steps.at(k_second_step).buffers_after.empty());
  REQUIRE(plan.steps.at(k_third_step).buffers_after.size() == 1);
  REQUIRE(plan.steps.at(k_fourth_step).buffers_before.size() == 1);
}

TEST_CASE("compute binding preserves a sampled descriptor's GENERAL layout", "[vkexec][resource-state][pass]")
{
  char image_storage{};
  auto *const image = fake_image(&image_storage);
  vkexec::handles::compute_pipeline pipe{};
  pipe.binding_count = k_one_binding_count;
  pipe.binding_accesses.push_back(vkexec::buffer_access::readonly);
  pipe.binding_kinds.push_back(vkexec::resource_kind::sampled_image);
  auto const table = vkexec::bindings(vkexec::resource_binding{ .slot = k_sampled_binding,
    .resource = vkexec::sampled_image_resource(
      image, VK_NULL_HANDLE, k_color_range, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) });
  auto const bind = vkexec::bind_compute(pipe, VK_NULL_HANDLE, table);
  REQUIRE(bind.metadata == vkexec::resource_metadata_state::complete);
  REQUIRE(bind.images.size() == k_single_step_count);

  vkexec::detail::resource_state_tracker tracker{ k_single_step_count };
  REQUIRE(tracker.use(bind.images.front(), k_first_step, k_default_family));
  auto const plan = std::move(tracker).finish();
  auto const &barrier = plan.steps.front().images_before.front();
  REQUIRE(barrier.old_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  REQUIRE(barrier.new_layout == VK_IMAGE_LAYOUT_GENERAL);
  REQUIRE(barrier.dst_access == VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

TEST_CASE("compute binding rejects missing storage buffer metadata", "[vkexec][pass]")
{
  constexpr std::uint32_t k_storage_binding = 0;
  vkexec::handles::compute_pipeline pipe{};
  pipe.binding_accesses = { vkexec::buffer_access::readonly };
  auto const table = vkexec::bindings(vkexec::resource_binding{
    .slot = k_storage_binding, .resource = vkexec::buffer_resource(VK_NULL_HANDLE, sizeof(std::uint32_t)) });
  auto const bind = vkexec::bind_compute(pipe, VK_NULL_HANDLE, table);
  REQUIRE(bind.metadata == vkexec::resource_metadata_state::partial);
}

TEST_CASE("opaque compute binding requires complete declarations before planning", "[vkexec][resource-state][pass]")
{
  char storage{};
  vkexec::handles::compute_pipeline pipe{};
  pipe.binding_accesses = { vkexec::buffer_access::readwrite };
  auto const opaque = vkexec::bind_compute(pipe);
  auto const queue = vkexec::queue_ref{ .family = k_default_family };
  vkexec::detail::resource_state_tracker tracker{ k_single_step_count };
  auto const missing =
    vkexec::detail::compute_pass_step<vkexec::detail::no_push_constants, vkexec::dispatch>{ .bind = opaque };
  REQUIRE_FALSE(vkexec::detail::declare_step_resources(missing, tracker, k_first_step, queue, false));

  auto const declared = vkexec::detail::compute_pass_step<vkexec::detail::no_push_constants, vkexec::dispatch>{
    .bind =
      opaque.with_resources({}, { vkexec::read_write(fake_buffer(&storage), vkexec::buffer_usage::storage_compute) })
  };
  REQUIRE(vkexec::detail::declare_step_resources(declared, tracker, k_first_step, queue, false));
}

TEST_CASE("descriptor-free compute binding remains complete", "[vkexec][resource-state][pass]")
{
  vkexec::handles::compute_pipeline const pipe{};
  auto const bind = vkexec::bind_compute(pipe);
  REQUIRE(bind.metadata == vkexec::resource_metadata_state::complete);
  auto const step =
    vkexec::detail::compute_pass_step<vkexec::detail::no_push_constants, vkexec::dispatch>{ .bind = bind };
  vkexec::detail::resource_state_tracker tracker{ k_single_step_count };
  REQUIRE(vkexec::detail::declare_step_resources(
    step, tracker, k_first_step, vkexec::queue_ref{ .family = k_default_family }, false));
}

TEST_CASE("binding count prevents incomplete compute metadata from appearing empty", "[vkexec][pass]")
{
  vkexec::handles::compute_pipeline pipe{};
  pipe.binding_count = k_one_binding_count;
  auto const raw = vkexec::bind_compute(pipe);
  REQUIRE(raw.metadata == vkexec::resource_metadata_state::none);
  auto const inferred = vkexec::bind_compute(pipe, VK_NULL_HANDLE, vkexec::resource_table{});
  REQUIRE(inferred.metadata == vkexec::resource_metadata_state::partial);
}

TEST_CASE("no-push classic overload accepts explicit opaque resources", "[vkexec][pass]")
{
  char storage{};
  vkexec::handles::compute_pipeline pipe{};
  pipe.binding_count = k_one_binding_count;
  auto expression =
    vkexec::detail::pass_fragment_root{}
    | vkexec::compute_pass(pipe,
      VK_NULL_HANDLE,
      std::vector<vkexec::image_use>{},
      std::vector<vkexec::buffer_use>{ vkexec::write(fake_buffer(&storage), vkexec::buffer_usage::storage_compute) },
      k_one_binding_count);
  auto chain = vkexec::detail::collect_pass_chain(std::move(expression), vkexec::scheduler_env{});
  vkexec::queue_ref const queue{ .family = k_default_family };
  auto execution = vkexec::detail::plan_dag(chain.step_queues, chain.step_predecessors, queue);
  REQUIRE(execution);
  REQUIRE(vkexec::detail::plan_resource_sync(*execution, chain.steps));
}

TEST_CASE("indirect dispatch tracks its buffer with explicit opaque declarations", "[vkexec][resource-state][pass]")
{
  char storage{};
  vkexec::handles::compute_pipeline pipe{};
  pipe.binding_accesses = { vkexec::buffer_access::readonly };
  auto const opaque = vkexec::bind_compute(pipe);
  auto const queue = vkexec::queue_ref{ .family = k_default_family };
  vkexec::detail::resource_state_tracker tracker{ k_pair_step_count };
  REQUIRE(tracker.use(
    vkexec::write(
      fake_buffer(&storage), VkDeviceSize{}, sizeof(VkDispatchIndirectCommand), vkexec::buffer_usage::storage_compute),
    k_first_step,
    k_default_family));
  auto const step = vkexec::detail::compute_pass_step<vkexec::detail::no_push_constants, vkexec::indirect_dispatch>{
    .bind = opaque.with_resources({}, {}), .dispatch_info = { .buffer = fake_buffer(&storage) }
  };
  REQUIRE(vkexec::detail::declare_step_resources(step, tracker, k_second_step, queue, false));
  auto const plan = std::move(tracker).finish();
  REQUIRE(std::ranges::find(plan.dependencies, std::pair{ k_first_step, k_second_step }) != plan.dependencies.end());
}

TEST_CASE("missing opaque metadata does not mutate indirect resource tracking", "[vkexec][resource-state][pass]")
{
  char storage{};
  vkexec::handles::compute_pipeline pipe{};
  pipe.binding_accesses = { vkexec::buffer_access::readonly };
  auto const step = vkexec::detail::compute_pass_step<vkexec::detail::no_push_constants, vkexec::indirect_dispatch>{
    .bind = vkexec::bind_compute(pipe), .dispatch_info = { .buffer = fake_buffer(&storage) }
  };
  vkexec::detail::resource_state_tracker tracker{ k_single_step_count };
  REQUIRE_FALSE(step.declare_resources(tracker, k_first_step, vkexec::queue_ref{ .family = k_default_family }));
  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.dependencies.empty());
  REQUIRE(plan.steps.front().buffers_before.empty());
}

TEST_CASE("partial compute metadata fails static and dynamic planning", "[vkexec][pass]")
{
  vkexec::handles::compute_pipeline pipe{};
  pipe.binding_accesses = { vkexec::buffer_access::readonly };
  auto const partial = vkexec::bind_compute(pipe, VK_NULL_HANDLE, vkexec::resource_table{});
  REQUIRE(partial.metadata == vkexec::resource_metadata_state::partial);
  auto const step =
    vkexec::detail::compute_pass_step<vkexec::detail::no_push_constants, vkexec::dispatch>{ .bind = partial };
  vkexec::detail::execution_plan execution{};
  execution.batches.push_back(vkexec::detail::execution_batch{ .queue = vkexec::queue_ref{ .family = k_default_family },
    .first_step = k_first_step,
    .end_step = k_single_step_count,
    .predecessors = {} });
  auto static_steps = std::tuple{ step };
  REQUIRE_FALSE(vkexec::detail::plan_resource_sync(execution, static_steps));
  std::vector<vkexec::detail::dynamic_pass_step> dynamic_steps;
  dynamic_steps.emplace_back(step);
  REQUIRE_FALSE(vkexec::detail::plan_resource_sync(execution, dynamic_steps));
}

TEST_CASE("when_all opaque branches require declarations and order conflicts", "[vkexec][pass]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::handles::compute_pipeline pipe{};
  pipe.binding_accesses = { vkexec::buffer_access::readwrite };
  auto const opaque = vkexec::bind_compute(pipe);
  auto const writer = opaque.with_resources(
    {}, { vkexec::write(buffer, VkDeviceSize{}, k_byte_size, vkexec::buffer_usage::storage_compute) });
  auto const reader = opaque.with_resources(
    {}, { vkexec::read(buffer, VkDeviceSize{}, k_byte_size, vkexec::buffer_usage::storage_compute) });
  auto make_root = []() -> decltype(auto) {
    return vkexec::make_pass_adaptor(vkexec::detail::make_callback_pass_step(
      [](vkexec::context &, VkCommandBuffer, vkexec::detail::pass_cleanup &) -> vkexec::status { return {}; }));
  };
  auto expression = vkexec::detail::pass_fragment_root{} | make_root()
                    | vkexec::when_all(vkexec::compute_pass(writer, vkexec::dispatch{}),
                      vkexec::compute_pass(reader, vkexec::dispatch{}));
  auto chain = vkexec::detail::collect_pass_chain(std::move(expression), vkexec::scheduler_env{});
  vkexec::queue_ref const queue{ .family = k_default_family };
  auto execution = vkexec::detail::plan_dag(chain.step_queues, chain.step_predecessors, queue);
  REQUIRE(execution);
  auto synchronization = vkexec::detail::plan_resource_sync(*execution, chain.steps);
  REQUIRE(synchronization);
  REQUIRE(std::ranges::find(synchronization->dependencies, std::pair{ k_second_step, k_third_step })
          != synchronization->dependencies.end());

  auto missing_expression = vkexec::detail::pass_fragment_root{} | make_root()
                            | vkexec::when_all(vkexec::compute_pass(opaque, vkexec::dispatch{}),
                              vkexec::compute_pass(reader, vkexec::dispatch{}));
  auto missing_chain = vkexec::detail::collect_pass_chain(std::move(missing_expression), vkexec::scheduler_env{});
  auto missing_execution = vkexec::detail::plan_dag(missing_chain.step_queues, missing_chain.step_predecessors, queue);
  REQUIRE(missing_execution);
  REQUIRE_FALSE(vkexec::detail::plan_resource_sync(*missing_execution, missing_chain.steps));
}

TEST_CASE("explicit opaque accesses generate cross-queue dependencies and ownership", "[vkexec][resource-state][pass]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::handles::compute_pipeline pipe{};
  pipe.binding_accesses = { vkexec::buffer_access::readwrite };
  auto const opaque = vkexec::bind_compute(pipe);
  using step_type = vkexec::detail::compute_pass_step<vkexec::detail::no_push_constants, vkexec::dispatch>;
  auto const write_step =
    step_type{ .bind = opaque.with_resources(
                 {}, { vkexec::write(buffer, VkDeviceSize{}, k_byte_size, vkexec::buffer_usage::storage_compute) }) };
  auto const read_step =
    step_type{ .bind = opaque.with_resources(
                 {}, { vkexec::read(buffer, VkDeviceSize{}, k_byte_size, vkexec::buffer_usage::storage_compute) }) };
  auto const rewrite_step =
    step_type{ .bind = opaque.with_resources(
                 {}, { vkexec::write(buffer, VkDeviceSize{}, k_byte_size, vkexec::buffer_usage::storage_compute) }) };
  vkexec::detail::resource_state_tracker tracker{ k_dag_step_count };
  REQUIRE(vkexec::detail::declare_step_resources(
    write_step, tracker, k_first_step, vkexec::queue_ref{ .family = k_default_family }, true));
  REQUIRE(vkexec::detail::declare_step_resources(
    read_step, tracker, k_second_step, vkexec::queue_ref{ .family = k_other_family }, true));
  REQUIRE(vkexec::detail::declare_step_resources(
    rewrite_step, tracker, k_third_step, vkexec::queue_ref{ .family = k_other_family }, true));
  REQUIRE(vkexec::detail::declare_step_resources(
    rewrite_step, tracker, k_fourth_step, vkexec::queue_ref{ .family = k_other_family }, true));
  auto const plan = std::move(tracker).finish();
  REQUIRE(std::ranges::find(plan.dependencies, std::pair{ k_first_step, k_second_step }) != plan.dependencies.end());
  REQUIRE(std::ranges::find(plan.dependencies, std::pair{ k_second_step, k_third_step }) != plan.dependencies.end());
  REQUIRE(std::ranges::find(plan.dependencies, std::pair{ k_third_step, k_fourth_step }) != plan.dependencies.end());
  REQUIRE_FALSE(plan.steps.at(k_first_step).buffers_after.empty());
  REQUIRE_FALSE(plan.steps.at(k_second_step).buffers_before.empty());
}

TEST_CASE("explicit opaque accesses order different queues in one family", "[vkexec][resource-state][pass]")
{
  char storage{};
  char first_queue_storage{};
  char second_queue_storage{};
  auto *const buffer = fake_buffer(&storage);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  auto *const first_queue = reinterpret_cast<VkQueue>(&first_queue_storage);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  auto *const second_queue = reinterpret_cast<VkQueue>(&second_queue_storage);
  vkexec::handles::compute_pipeline pipe{};
  pipe.binding_accesses = { vkexec::buffer_access::readwrite };
  auto const opaque = vkexec::bind_compute(pipe);
  using step_type = vkexec::detail::compute_pass_step<vkexec::detail::no_push_constants, vkexec::dispatch>;
  auto const writer =
    step_type{ .bind = opaque.with_resources(
                 {}, { vkexec::write(buffer, VkDeviceSize{}, k_byte_size, vkexec::buffer_usage::storage_compute) }) };
  auto const reader =
    step_type{ .bind = opaque.with_resources(
                 {}, { vkexec::read(buffer, VkDeviceSize{}, k_byte_size, vkexec::buffer_usage::storage_compute) }) };
  vkexec::detail::resource_state_tracker tracker{ k_pair_step_count };
  REQUIRE(vkexec::detail::declare_step_resources(
    writer, tracker, k_first_step, vkexec::queue_ref{ .queue = first_queue, .family = k_default_family }, false));
  REQUIRE(vkexec::detail::declare_step_resources(
    reader, tracker, k_second_step, vkexec::queue_ref{ .queue = second_queue, .family = k_default_family }, false));
  auto const plan = std::move(tracker).finish();
  REQUIRE(std::ranges::find(plan.dependencies, std::pair{ k_first_step, k_second_step }) != plan.dependencies.end());
}

TEST_CASE("resource tracking plans cross family release and acquire", "[vkexec][resource-state][pass]")
{
  char storage{};
  constexpr std::uint32_t k_graphics_family = 2;
  constexpr std::uint32_t k_compute_family = 3;
  vkexec::detail::resource_state_tracker tracker{ k_pair_step_count };
  REQUIRE(tracker.use(vkexec::write(fake_image(&storage), k_color_range, vkexec::image_usage::color_attachment),
    k_first_step,
    k_graphics_family));
  REQUIRE(tracker.use(vkexec::read(fake_image(&storage), k_color_range, vkexec::image_usage::sampled_compute),
    k_second_step,
    k_compute_family));

  auto const plan = std::move(tracker).finish();
  REQUIRE(plan.steps.at(k_first_step).images_after.size() == 1);
  REQUIRE(plan.steps.at(k_second_step).images_before.size() == 1);
  auto const &release = plan.steps.at(k_first_step).images_after.front();
  auto const &acquire = plan.steps.at(k_second_step).images_before.front();
  REQUIRE(release.src_queue_family == k_graphics_family);
  REQUIRE(release.dst_queue_family == k_compute_family);
  REQUIRE(release.dst_stage == VK_PIPELINE_STAGE_2_NONE);
  REQUIRE(acquire.src_stage == VK_PIPELINE_STAGE_2_NONE);
  REQUIRE(acquire.dst_stage == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  REQUIRE(release.new_layout == acquire.new_layout);
}

TEST_CASE("resource tracking requires an initial layout for first reads",
  "[vkexec][resource-state][pass][resource-helpers]")
{
  char storage{};
  vkexec::detail::resource_state_tracker tracker{ k_single_step_count };
  REQUIRE_FALSE(tracker.use(vkexec::read(fake_image(&storage), k_color_range, vkexec::image_usage::sampled_compute),
    k_first_step,
    k_default_family));
  REQUIRE(tracker.use(vkexec::read(fake_image(&storage),
                        k_color_range,
                        vkexec::image_usage::sampled_compute,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
    k_first_step,
    k_default_family));
}

TEST_CASE("resource tracking uses semaphore ordering across queues in one family", "[vkexec][resource-state][pass]")
{
  char image_storage{};
  char first_queue_storage{};
  char second_queue_storage{};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  auto *first_queue = reinterpret_cast<VkQueue>(&first_queue_storage);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- opaque test-only queue handle
  auto *second_queue = reinterpret_cast<VkQueue>(&second_queue_storage);
  vkexec::detail::resource_state_tracker tracker{ k_pair_step_count };
  REQUIRE(tracker.use(vkexec::write(fake_image(&image_storage), k_color_range, vkexec::image_usage::color_attachment),
    k_first_step,
    k_default_family,
    first_queue));
  REQUIRE(tracker.use(vkexec::read(fake_image(&image_storage), k_color_range, vkexec::image_usage::sampled_compute),
    k_second_step,
    k_default_family,
    second_queue));

  auto const plan = std::move(tracker).finish();
  auto const &barrier = plan.steps.at(k_second_step).images_before.front();
  REQUIRE(barrier.src_stage == VK_PIPELINE_STAGE_2_NONE);
  REQUIRE(barrier.src_access == VK_ACCESS_2_NONE);
  REQUIRE(barrier.dst_stage == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  REQUIRE(barrier.src_queue_family == VK_QUEUE_FAMILY_IGNORED);
}

TEST_CASE("resource tracking rejects access incompatible with image usage", "[vkexec][resource-state][pass]")
{
  char storage{};
  vkexec::detail::resource_state_tracker tracker{ k_single_step_count };
  auto *const image = fake_image(&storage);
  REQUIRE_FALSE(tracker.use(
    vkexec::write(image, k_color_range, vkexec::image_usage::sampled_fragment), k_first_step, k_default_family));
  REQUIRE_FALSE(tracker.use(
    vkexec::read(image, k_color_range, vkexec::image_usage::transfer_destination), k_first_step, k_default_family));
}

TEST_CASE("resource tracking rejects access incompatible with buffer usage", "[vkexec][resource-state][pass]")
{
  char storage{};
  vkexec::detail::resource_state_tracker tracker{ k_single_step_count };
  auto *const buffer = fake_buffer(&storage);
  REQUIRE_FALSE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::vertex), k_first_step, k_default_family));
  REQUIRE_FALSE(
    tracker.use(vkexec::read(buffer, vkexec::buffer_usage::transfer_destination), k_first_step, k_default_family));
}

TEST_CASE("uniform buffer usage selects one shader stage", "[vkexec][pass]")
{
  char compute_storage{};
  char vertex_storage{};
  auto const compute = vkexec::read(fake_buffer(&compute_storage), vkexec::buffer_usage::uniform_compute);
  auto const vertex = vkexec::read(fake_buffer(&vertex_storage), vkexec::buffer_usage::uniform_vertex);
  REQUIRE(vkexec::detail::buffer_scope(compute).stage == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  REQUIRE(vkexec::detail::buffer_scope(vertex).stage == VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
}

TEST_CASE("resource tracking rejects two declarations in one pass", "[vkexec][resource-state][pass]")
{
  char image_storage{};
  char buffer_storage{};
  auto *const image = fake_image(&image_storage);
  auto *const buffer = fake_buffer(&buffer_storage);
  vkexec::detail::resource_state_tracker tracker{ k_single_step_count };
  REQUIRE(tracker.use(
    vkexec::write(image, k_color_range, vkexec::image_usage::color_attachment), k_first_step, k_default_family));
  REQUIRE_FALSE(tracker.use(
    vkexec::read(image, k_color_range, vkexec::image_usage::sampled_fragment), k_first_step, k_default_family));
  REQUIRE(tracker.use(vkexec::write(buffer, vkexec::buffer_usage::storage_compute), k_first_step, k_default_family));
  REQUIRE_FALSE(
    tracker.use(vkexec::read(buffer, vkexec::buffer_usage::storage_compute), k_first_step, k_default_family));
}

TEST_CASE("presentation validation checks the declared image subresources", "[vkexec][resource-state][pass]")
{
  char storage{};
  auto *const image = fake_image(&storage);
  vkexec::detail::execution_plan execution{};
  execution.batches.push_back(vkexec::detail::execution_batch{
    .queue = {}, .first_step = k_first_step, .end_step = k_single_step_count, .predecessors = {} });
  auto declaration = vkexec::write(image, k_color_range, vkexec::image_usage::color_attachment);
  declaration.final_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  vkexec::detail::resource_state_tracker tracker{ k_single_step_count };
  REQUIRE(tracker.use(declaration, k_first_step, k_default_family));
  auto plan = std::move(tracker).finish();
  REQUIRE(vkexec::detail::validate_presentation_image(plan, execution, image));
  plan.image_uses.front().range.baseMipLevel = 1;
  REQUIRE_FALSE(vkexec::detail::validate_presentation_image(plan, execution, image));
  plan.image_uses.front().range = k_color_range;
  plan.image_uses.front().range.baseArrayLayer = 1;
  REQUIRE_FALSE(vkexec::detail::validate_presentation_image(plan, execution, image));
}

TEST_CASE("resource planning keeps disjoint dynamic graph branches independent", "[vkexec][pass][ranges]")
{
  char storage{};
  auto *const buffer = fake_buffer(&storage);
  vkexec::dynamic_pass_graph_sender graph{};
  auto const record = [](VkCommandBuffer) -> void {};
  auto const first = graph.append_after({},
    vkexec::custom_pass(
      vkexec::uses(vkexec::write(buffer, 0, k_byte_size, vkexec::buffer_usage::storage_compute)), record));
  auto const second = graph.append_after({},
    vkexec::custom_pass(
      vkexec::uses(vkexec::write(buffer, k_byte_size, k_byte_size, vkexec::buffer_usage::storage_compute)), record));
  vkexec::queue_ref const queue{ .family = k_default_family };
  auto execution = vkexec::detail::plan_dag(graph.step_queues, graph.step_predecessors, queue);
  REQUIRE(execution);
  auto synchronization = vkexec::detail::plan_resource_sync(*execution, graph.steps);
  REQUIRE(synchronization);
  REQUIRE(synchronization->dependencies.empty());
  vkexec::detail::add_resource_dependencies(*execution, *synchronization);
  REQUIRE(execution->batches.at(first).predecessors.empty());
  REQUIRE(execution->batches.at(second).predecessors.empty());
}


// NOLINTEND(readability-function-cognitive-complexity)
