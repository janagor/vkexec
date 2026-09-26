#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vkexec/error.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>
#include <vkexec/sync_wait.hpp>

#include <stdexec/execution.hpp>

#include <concepts>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace {

constexpr auto k_expected_value = 42;
constexpr auto k_sync_wait_pair_integer = 1;
constexpr auto k_sync_wait_pair_floating_point = 2.0;

struct status_factory
{
  auto operator()() const -> vkexec::status { return {}; }
};

struct empty_factory
{
  auto operator()() const -> vkexec::result<int> { return k_expected_value; }
};

struct move_only_factory
{
  std::unique_ptr<int> value;

  explicit move_only_factory(std::unique_ptr<int> initial_value) : value(std::move(initial_value)) {}
  ~move_only_factory() = default;
  move_only_factory(move_only_factory &&) = default;
  move_only_factory(move_only_factory const &) = delete;
  auto operator=(move_only_factory &&) -> move_only_factory & = default;
  auto operator=(move_only_factory const &) -> move_only_factory & = delete;

  auto operator()() const -> vkexec::result<int> { return *value; }
};

struct copyable_factory
{
  int value{ k_expected_value };
  auto operator()() const -> vkexec::result<int> { return value; }
};

struct move_construct_only
{
  int value{};

  move_construct_only() = default;
  explicit move_construct_only(int initial_value) : value(initial_value) {}
  ~move_construct_only() = default;

  move_construct_only(move_construct_only &&) = default;
  move_construct_only(move_construct_only const &) = delete;

  auto operator=(move_construct_only &&) -> move_construct_only & = delete;
  auto operator=(move_construct_only const &) -> move_construct_only & = delete;
};

struct reference_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t(int &)>;
};

struct const_reference_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t(int const &)>;
};

struct multiple_value_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t(int), stdexec::set_value_t(double)>;
};

struct no_value_sender
{
  using sender_concept = stdexec::sender_t;
  using completion_signatures = stdexec::completion_signatures<stdexec::set_error_t(vkexec::error)>;
};

}// namespace

template<class Sender>
concept sync_wait_value_tuple_well_formed = requires { typename vkexec::detail::sync_wait_value_tuple_t<Sender>; };

static_assert(stdexec::sender<vkexec::factory_sender<empty_factory>>);
static_assert(stdexec::sender<vkexec::factory_sender<status_factory>>);

static_assert(std::same_as<vkexec::factory_sender<empty_factory>::completion_signatures,
  stdexec::
    completion_signatures<stdexec::set_value_t(int), stdexec::set_error_t(vkexec::error), stdexec::set_stopped_t()>>);

static_assert(std::same_as<vkexec::factory_sender<status_factory>::completion_signatures,
  stdexec::
    completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(vkexec::error), stdexec::set_stopped_t()>>);

static_assert(std::move_constructible<vkexec::factory_sender<move_only_factory>>);
static_assert(!std::copy_constructible<vkexec::factory_sender<move_only_factory>>);
static_assert(stdexec::sender<vkexec::factory_sender<move_only_factory>>);
static_assert(std::copy_constructible<vkexec::factory_sender<copyable_factory>>);
static_assert(std::move_constructible<move_construct_only>);
static_assert(!std::is_move_assignable_v<move_construct_only>);

using sync_wait_empty_sender = decltype(stdexec::just());
using sync_wait_int_sender = decltype(stdexec::just(k_expected_value));
using sync_wait_pair_sender = decltype(stdexec::just(k_sync_wait_pair_integer, k_sync_wait_pair_floating_point));

static_assert(std::same_as<vkexec::detail::sync_wait_value_tuple_t<sync_wait_empty_sender>, std::tuple<>>);
static_assert(std::same_as<vkexec::detail::sync_wait_value_tuple_t<sync_wait_int_sender>, std::tuple<int>>);
static_assert(std::same_as<vkexec::detail::sync_wait_value_tuple_t<sync_wait_pair_sender>, std::tuple<int, double>>);
static_assert(std::same_as<vkexec::detail::sync_wait_value_tuple_t<reference_sender>, std::tuple<int>>);
static_assert(std::same_as<vkexec::detail::sync_wait_value_tuple_t<const_reference_sender>, std::tuple<int>>);
static_assert(stdexec::sender_in<sync_wait_int_sender, vkexec::detail::sync_wait_env>);
static_assert(stdexec::sender_to<sync_wait_int_sender, vkexec::detail::sync_wait_receiver_t<sync_wait_int_sender>>);
static_assert(!sync_wait_value_tuple_well_formed<multiple_value_sender>);
static_assert(!sync_wait_value_tuple_well_formed<no_value_sender>);

TEST_CASE("factory_sender preserves concrete callable type", "[vkexec][sender]")
{
  auto lambda = []() -> vkexec::result<int> { return k_expected_value; };
  auto sndr = vkexec::make_sender(lambda);

  STATIC_REQUIRE(std::same_as<decltype(sndr), vkexec::factory_sender<decltype(lambda)>>);
  REQUIRE(vkexec::try_sync_wait_value(sndr).has_value());
}

TEST_CASE("factory_sender completes with a value", "[vkexec][sender]")
{
  auto outcome =
    vkexec::try_sync_wait_value(vkexec::make_sender([]() -> vkexec::result<int> { return k_expected_value; }));
  REQUIRE(outcome.has_value());
  REQUIRE(vkexec::expected_take(outcome) == k_expected_value);
}

TEST_CASE("factory_sender completes with an error", "[vkexec][sender]")
{
  auto outcome = vkexec::try_sync_wait(
    vkexec::make_sender([]() -> vkexec::result<int> { return vkexec::fail(vkexec::errc::unsupported); }));
  REQUIRE(outcome.failed());
  REQUIRE(outcome.take_error().code == vkexec::make_error_code(vkexec::errc::unsupported));
}

TEST_CASE("factory_sender status completes successfully", "[vkexec][sender]")
{
  auto outcome = vkexec::try_sync_wait(vkexec::make_sender([]() -> vkexec::status { return {}; }));
  REQUIRE(outcome.has_value());
}

TEST_CASE("factory_sender status completes with an error", "[vkexec][sender]")
{
  auto outcome = vkexec::try_sync_wait(
    vkexec::make_sender([]() -> vkexec::status { return vkexec::fail(vkexec::errc::invalid_argument); }));
  REQUIRE(outcome.failed());
  REQUIRE(outcome.take_error().code == vkexec::make_error_code(vkexec::errc::invalid_argument));
}

TEST_CASE("factory_sender respects pre-requested stop", "[vkexec][sender]")
{
  bool invoked = false;
  stdexec::inplace_stop_source source;
  source.request_stop();

  auto stopped_sndr = stdexec::write_env(vkexec::make_sender([&invoked]() -> vkexec::result<int> {
    invoked = true;
    return k_expected_value;
  }),
    stdexec::prop{ stdexec::get_stop_token, source.get_token() });

  auto const waited = vkexec::test::sync_wait_sender(std::move(stopped_sndr));
  REQUIRE_FALSE(invoked);
  REQUIRE(vkexec::test::sync_wait_stopped(waited));
}

TEST_CASE("factory_sender supports move-only factories", "[vkexec][sender]")
{
  auto ptr = std::make_unique<int>(k_expected_value);
  auto sndr = vkexec::make_sender([ptr = std::move(ptr)]() mutable -> vkexec::result<int> { return *ptr; });

  auto outcome = vkexec::try_sync_wait_value(std::move(sndr));
  REQUIRE(outcome.has_value());
  REQUIRE(vkexec::expected_take(outcome) == k_expected_value);
}

TEST_CASE("factory_sender does not require move-assignable values", "[vkexec][sender]")
{
  auto sndr = vkexec::make_sender(
    []() -> vkexec::result<move_construct_only> { return move_construct_only{ k_expected_value }; });

  auto outcome = vkexec::try_sync_wait_value(sndr);
  REQUIRE(outcome.has_value());
  REQUIRE(vkexec::expected_take(outcome).value == k_expected_value);
}

TEST_CASE("factory_sender copyable factory supports lvalue connect", "[vkexec][sender]")
{
  auto sndr = vkexec::make_sender(copyable_factory{});
  auto outcome = vkexec::try_sync_wait_value(sndr);
  REQUIRE(outcome.has_value());
  REQUIRE(vkexec::expected_take(outcome) == k_expected_value);

  auto again = vkexec::try_sync_wait_value(sndr);
  REQUIRE(again.has_value());
  REQUIRE(vkexec::expected_take(again) == k_expected_value);
}

#if VKEXEC_ENABLE_EXCEPTIONS
TEST_CASE("factory_sender maps factory exceptions to set_error", "[vkexec][sender]")
{
  auto outcome =
    vkexec::try_sync_wait(vkexec::make_sender([]() -> vkexec::result<int> { throw std::runtime_error("boom"); }));
  REQUIRE(outcome.failed());
  REQUIRE(outcome.take_error().code == vkexec::unexpected_exception_error().code);
}
#endif
