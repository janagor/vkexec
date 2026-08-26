#include <catch2/catch_test_macros.hpp>

#include <vkexec/buffer.hpp>
#include <vkexec/bulk.hpp>
#include <vkexec/config.hpp>
#include <vkexec/context.hpp>
#include <vkexec/submit.hpp>
#include <vkexec/submit_scope.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec_edsl/push_constant.hpp>
#include <vkexec_edsl/trace.hpp>
#include <vkexec_edsl/types.hpp>

#include <boost/describe/class.hpp>

#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>
#include <vulkan/vulkan_core.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>

namespace ex = stdexec;
namespace edsl = vkexec::edsl;

namespace {

constexpr std::uint32_t k_work_count = 10;
constexpr float k_initial = 3.0F;
constexpr float k_add = 5.0F;
constexpr float k_epsilon = 1.0E-4F;

struct bulk_params
{
  float value;
};
// cppcheck-suppress unknownMacro
BOOST_DESCRIBE_STRUCT(bulk_params, (), (value))

struct nop_params
{
  float n;
};
// cppcheck-suppress unknownMacro
BOOST_DESCRIBE_STRUCT(nop_params, (), (n))

auto skip_if_no_vulkan(vkexec::error const &err) -> void
{ SKIP(std::string("Vulkan unavailable: ") + std::string(err.message())); }

}// namespace

TEST_CASE("write_traced_descriptors returns when no buffers are bound", "[vkexec][bulk]")
{ vkexec::detail::write_traced_descriptors(VK_NULL_HANDLE, VK_NULL_HANDLE, std::span<edsl::storage_trace const>{}); }

TEST_CASE("bulk factory stores shape and params", "[vkexec][bulk]")
{
  bulk_params const params{ .value = k_add };
  auto const closure =
    vkexec::bulk(k_work_count, params, [](edsl::Int /*idx*/, edsl::push_constant<bulk_params> /*push*/) -> void {});

  REQUIRE(closure.shape == k_work_count);
  REQUIRE(closure.params.value == k_add);
}

TEST_CASE("bulk kernel updates host-visible buffers", "[vkexec][bulk][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto values_result = vkexec::buffer<float>::create_sync(ctx, k_work_count, k_initial);
  REQUIRE(values_result.has_value());
  auto &values = *values_result;

  auto pipeline =
    ex::schedule(ctx.get_scheduler())
    | vkexec::bulk(
      k_work_count, bulk_params{ .value = k_add }, [&](edsl::Int idx, edsl::push_constant<bulk_params> push) -> void {
        values[idx] = values[idx] + push.get<&bulk_params::value>();
      });
  auto waited = vkexec::sync_wait(pipeline);
  REQUIRE(waited.has_value());
  REQUIRE(waited->has_value());

  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (std::uint32_t index = 0; index < k_work_count; ++index) {
    REQUIRE(std::fabs(values.data()[index] - (k_initial + k_add)) <= k_epsilon);
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

TEST_CASE("bulk traces kernels with no storage buffers", "[vkexec][bulk][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto sender =
    ex::schedule(ctx.get_scheduler())
    | vkexec::bulk(
      k_work_count, nop_params{ .n = k_add }, [](edsl::Int idx, edsl::push_constant<nop_params> push) -> void {
        edsl::Float const unused = edsl::Float::constant(0.0) * push.get<&nop_params::n>();
        (void)idx;
        (void)unused;
      });

  REQUIRE(sender.shape == k_work_count);
  auto waited = vkexec::sync_wait(sender);
  REQUIRE(waited.has_value());
  REQUIRE(waited->has_value());
}

TEST_CASE("submit sender completes after GPU work", "[vkexec][bulk][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto values_result = vkexec::buffer<float>::create_sync(ctx, k_work_count, k_initial);
  REQUIRE(values_result.has_value());
  auto &values = *values_result;

  auto pipeline = ex::schedule(ctx.get_scheduler())
                  | vkexec::bulk(k_work_count,
                    bulk_params{ .value = k_add },
                    [&](edsl::Int idx, edsl::push_constant<bulk_params> push) -> void {
                      values[idx] = values[idx] + push.get<&bulk_params::value>();
                    })
                  | vkexec::submit;
  auto waited = vkexec::sync_wait(pipeline);
  REQUIRE(waited.has_value());
  REQUIRE(waited->has_value());

  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  REQUIRE(std::fabs(values.data()[0] - (k_initial + k_add)) <= k_epsilon);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

TEST_CASE("submit overlaps two GPU dispatches via when_all", "[vkexec][bulk][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto left_result = vkexec::buffer<float>::create_sync(ctx, k_work_count, k_initial);
  REQUIRE(left_result.has_value());
  auto right_result = vkexec::buffer<float>::create_sync(ctx, k_work_count, k_initial);
  REQUIRE(right_result.has_value());
  auto &left = *left_result;
  auto &right = *right_result;

  auto make_async = [&](vkexec::buffer<float> &values) -> auto {
    return ex::schedule(ctx.get_scheduler())
           | vkexec::bulk(k_work_count,
             bulk_params{ .value = k_add },
             [&](edsl::Int idx, edsl::push_constant<bulk_params> push) -> void {
               values[idx] = values[idx] + push.get<&bulk_params::value>();
             })
           | vkexec::submit;
  };

  auto waited = vkexec::sync_wait(ex::when_all(make_async(left), make_async(right)));
  REQUIRE(waited.has_value());
  REQUIRE(waited->has_value());

  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  REQUIRE(std::fabs(left.data()[0] - (k_initial + k_add)) <= k_epsilon);
  REQUIRE(std::fabs(right.data()[0] - (k_initial + k_add)) <= k_epsilon);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

TEST_CASE("submit completes with set_stopped when stop is already requested", "[vkexec][bulk]")
{
  vkexec::scheduler const sched{ nullptr };
  ex::inplace_stop_source source;
  source.request_stop();

  auto sender = ex::schedule(sched)
                | vkexec::bulk(k_work_count,
                  bulk_params{ .value = k_add },
                  [](edsl::Int /*idx*/, edsl::push_constant<bulk_params> /*push*/) -> void {})
                | vkexec::submit;

  auto const waited =
    // NOLINTNEXTLINE(misc-include-cleaner)
    vkexec::sync_wait(ex::write_env(sender, ex::prop{ ex::get_stop_token, source.get_token() }));
  REQUIRE(waited.has_value());
  REQUIRE_FALSE(waited->has_value());
}

TEST_CASE("submit reclaims resources when stop races with GPU completion", "[vkexec][bulk][gpu]")
{
  auto ctx_result = vkexec::context::create();
  if (!ctx_result) { skip_if_no_vulkan(vkexec::to_error(ctx_result.error())); }
  auto &ctx = **ctx_result;

  auto values_result = vkexec::buffer<float>::create_sync(ctx, k_work_count, k_initial);
  REQUIRE(values_result.has_value());
  auto &values = *values_result;

  ex::inplace_stop_source source;
  auto pipeline = ex::schedule(ctx.get_scheduler())
                  | vkexec::bulk(k_work_count,
                    bulk_params{ .value = k_add },
                    [&](edsl::Int idx, edsl::push_constant<bulk_params> push) -> void {
                      values[idx] = values[idx] + push.get<&bulk_params::value>();
                    })
                  | vkexec::submit;

  auto env_sender =
    // NOLINTNEXTLINE(misc-include-cleaner)
    ex::write_env(pipeline, ex::prop{ ex::get_stop_token, source.get_token() });

  std::jthread const stopper{ [&source]() -> void { source.request_stop(); } };

  // May complete with value or stopped depending on timing; reclaim must not leak either way.
  (void)vkexec::sync_wait(std::move(env_sender));
}
