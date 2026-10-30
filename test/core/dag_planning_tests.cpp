#include <catch2/catch_test_macros.hpp>
#include <vkexec/detail/pass_execution.hpp>

#include <array>
#include <cstddef>
#include <vector>

namespace {
using vkexec::detail::execution_batch;
using vkexec::detail::execution_plan;
using edges = std::vector<std::vector<std::size_t>>;

struct queues_fixture
{
  char first_storage{};
  char second_storage{};
  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  vkexec::queue_ref first{ .queue = reinterpret_cast<VkQueue>(&first_storage), .family = 3 };
  vkexec::queue_ref second{ .queue = reinterpret_cast<VkQueue>(&second_storage), .family = 7 };
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
};

auto check_batch(execution_batch const &batch,
  vkexec::queue_ref queue,
  std::size_t first,
  std::size_t end,
  std::vector<std::size_t> const &incoming) -> void
{
  CHECK(batch.queue.queue == queue.queue);
  CHECK(batch.queue.family == queue.family);
  CHECK(batch.first_step == first);
  CHECK(batch.end_step == end);
  CHECK(batch.predecessors == incoming);
}

auto make_plan(edges const &incoming) -> execution_plan
{
  execution_plan plan;
  for (std::size_t i{}; i < incoming.size(); ++i) {
    plan.batches.push_back({ .queue = {}, .first_step = i, .end_step = i + 1, .predecessors = incoming.at(i) });
  }
  return plan;
}
}// namespace

TEST_CASE("DAG planning preserves exact nodes queues and edges", "[vkexec][pass][dag-plan]")
{
  queues_fixture q;
  std::vector<edges> const shapes{
    {}, { {} }, { {}, {}, {}, {} }, { {}, { 0 }, { 1 }, { 2 } }, { {}, { 0 }, { 0 }, { 1, 2 } }, { {}, {} }
  };
  for (auto const &incoming : shapes) {
    CAPTURE(incoming);
    std::vector<vkexec::queue_affinity> queues(incoming.size());
    if (queues.size() > 2) { queues.at(2) = q.second; }
    auto plan = vkexec::detail::plan_dag(queues, incoming, q.first);
    REQUIRE(plan);
    REQUIRE(plan->batches.size() == incoming.size());
    for (std::size_t i{}; i < incoming.size(); ++i) {
      check_batch(plan->batches.at(i), i == 2 ? q.second : q.first, i, i + 1, incoming.at(i));
    }
    REQUIRE(vkexec::detail::validate_execution_batches(*plan, incoming.size()));
  }
  std::array<vkexec::queue_affinity, 3> const queues{};
  REQUIRE_FALSE(vkexec::detail::plan_dag(queues, edges(2), q.first));
  std::vector<edges> const invalid{
    { {}, {}, { 2 } }, { {}, { 2 }, {} }, { {}, {}, {}, { 1, 1 } }, { {}, {}, { 10 } }, { { 0 } }
  };
  for (auto const &incoming : invalid) {
    CAPTURE(incoming);
    REQUIRE_FALSE(vkexec::detail::plan_dag(std::vector<vkexec::queue_affinity>(incoming.size()), incoming, q.first));
  }
}

TEST_CASE("terminal batches include every disconnected sink in index order", "[vkexec][pass][dag-plan]")
{
  std::vector<std::pair<edges, std::vector<std::size_t>>> const cases{ { {}, {} },
    { { {} }, { 0 } },
    { { {}, { 0 }, { 1 } }, { 2 } },
    { { {}, {} }, { 0, 1 } },
    { { {}, { 0 }, { 0 }, { 1, 2 } }, { 3 } },
    { { {}, { 0 }, {} }, { 1, 2 } } };
  for (auto const &[incoming, expected] : cases) {
    CAPTURE(incoming);
    CHECK(vkexec::detail::terminal_batches(make_plan(incoming)) == expected);
  }
}

TEST_CASE("execution batch validation rejects malformed partitions and edges", "[vkexec][pass][dag-plan]")
{
  struct validation_case
  {
    execution_plan plan;
    std::size_t steps;
    bool valid;
  };
  auto batch = [](std::size_t first, std::size_t end, std::vector<std::size_t> incoming = {}) -> execution_batch {
    return { .queue = {}, .first_step = first, .end_step = end, .predecessors = std::move(incoming) };
  };
  std::vector<validation_case> const cases{ { {}, 0, true },
    { {}, 1, false },
    { { { batch(0, 3) } }, 3, true },
    { { { batch(0, 2), batch(2, 4) } }, 4, true },
    { { { batch(0, 1), batch(2, 4) } }, 4, false },
    { { { batch(0, 3), batch(2, 4) } }, 4, false },
    { { { batch(0, 0) } }, 0, false },
    { { { batch(0, 2), batch(2, 2) } }, 2, false },
    { { { batch(1, 2) } }, 2, false },
    { { { batch(0, 3), batch(3, 2) } }, 3, false },
    { { { batch(0, 4) } }, 3, false },
    { { { batch(0, 2) } }, 3, false },
    { { { batch(0, 1, { 0 }) } }, 1, false },
    { { { batch(0, 1, { 1 }), batch(1, 2) } }, 2, false },
    { { { batch(0, 1), batch(1, 2, { 0 }) } }, 2, true } };
  for (std::size_t i{}; i < cases.size(); ++i) {
    CAPTURE(i);
    auto const &entry = cases.at(i);
    CHECK(vkexec::detail::validate_execution_batches(entry.plan, entry.steps).has_value() == entry.valid);
  }
}

TEST_CASE("resource hazards map steps to distinct batch predecessors in insertion order", "[vkexec][pass][dag-plan]")
{
  execution_plan const original{ { { .queue = {}, .first_step = 0, .end_step = 2, .predecessors = {} },
    { .queue = {}, .first_step = 2, .end_step = 4, .predecessors = {} },
    { .queue = {}, .first_step = 4, .end_step = 5, .predecessors = {} } } };
  using dependency = std::pair<std::size_t, std::size_t>;
  struct dependency_case
  {
    std::vector<dependency> dependencies;
    edges initial;
    edges expected;
  };
  std::vector<dependency_case> const cases{ { {}, { {}, {}, {} }, { {}, {}, {} } },
    { { { 0, 1 } }, { {}, {}, {} }, { {}, {}, {} } },
    { { { 0, 2 }, { 1, 3 } }, { {}, {}, {} }, { {}, { 0 }, {} } },
    { { { 0, 2 } }, { {}, { 0 }, {} }, { {}, { 0 }, {} } },
    { { { 2, 4 }, { 0, 4 } }, { {}, {}, {} }, { {}, {}, { 1, 0 } } },
    { { { 0, 1 }, { 0, 2 }, { 1, 3 }, { 2, 4 }, { 3, 4 }, { 1, 4 } }, { {}, {}, {} }, { {}, { 0 }, { 1, 0 } } } };
  for (auto const &entry : cases) {
    CAPTURE(entry.dependencies);
    auto plan = original;
    for (std::size_t i{}; i < plan.batches.size(); ++i) { plan.batches.at(i).predecessors = entry.initial.at(i); }
    REQUIRE(vkexec::detail::validate_execution_batches(plan, 5));
    vkexec::detail::resource_sync_plan sync;
    sync.steps.resize(5);
    sync.dependencies = entry.dependencies;
    vkexec::detail::add_resource_dependencies(plan, sync);
    REQUIRE(plan.batches.size() == original.batches.size());
    for (std::size_t i{}; i < plan.batches.size(); ++i) {
      auto const &before = original.batches.at(i);
      check_batch(plan.batches.at(i), before.queue, before.first_step, before.end_step, entry.expected.at(i));
    }
    REQUIRE(vkexec::detail::validate_execution_batches(plan, 5));
  }
}

TEST_CASE("sequential planning resolves absent affinities and compares both queue fields", "[vkexec][pass][dag-plan]")
{
  queues_fixture q;
  auto empty = vkexec::detail::plan_execution(0, {}, q.first);
  REQUIRE(empty);
  CHECK(empty->batches.empty());
  auto defaults = vkexec::detail::plan_execution(3, {}, q.first);
  REQUIRE(defaults);
  REQUIRE(defaults->batches.size() == 1);
  check_batch(defaults->batches.front(), q.first, 0, 3, {});
  vkexec::queue_ref const other_family{ .queue = q.first.queue, .family = q.second.family };
  vkexec::queue_ref const other_handle{ .queue = q.second.queue, .family = q.first.family };
  std::array<vkexec::queue_affinity, 4> const queues{ q.first, other_family, other_handle, other_handle };
  auto planned = vkexec::detail::plan_execution(queues.size(), queues, q.first);
  REQUIRE(planned);
  REQUIRE(planned->batches.size() == 3);
  check_batch(planned->batches.at(0), q.first, 0, 1, {});
  check_batch(planned->batches.at(1), other_family, 1, 2, { 0 });
  check_batch(planned->batches.at(2), other_handle, 2, 4, { 1 });
  REQUIRE_FALSE(vkexec::detail::plan_execution(0, queues, q.first));
  REQUIRE_FALSE(vkexec::detail::plan_execution(queues.size() + 1, queues, q.first));
}

TEST_CASE("queue family crossing depends on families rather than queue handles", "[vkexec][pass][dag-plan]")
{
  queues_fixture q;
  execution_plan plan;
  CHECK_FALSE(vkexec::detail::crosses_queue_families(plan));
  plan.batches.push_back({ .queue = q.first, .first_step = 0, .end_step = 1, .predecessors = {} });
  CHECK_FALSE(vkexec::detail::crosses_queue_families(plan));
  plan.batches.push_back({ .queue = { .queue = q.second.queue, .family = q.first.family },
    .first_step = 1,
    .end_step = 2,
    .predecessors = {} });
  CHECK_FALSE(vkexec::detail::crosses_queue_families(plan));
  plan.batches.push_back({ .queue = q.second, .first_step = 2, .end_step = 3, .predecessors = {} });
  CHECK(vkexec::detail::crosses_queue_families(plan));
}
