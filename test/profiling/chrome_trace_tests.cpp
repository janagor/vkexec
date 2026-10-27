#include <catch2/catch_test_macros.hpp>

#include <vkexec_profiling/chrome_trace.hpp>
#include <vkexec_profiling/profile_event.hpp>

#include <cstdint>
#include <ios>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {
using vkexec::profiling::profile_event;
using vkexec::profiling::sample_status;
using vkexec::profiling::write_chrome_trace;

constexpr std::uint32_t k_sixteen_bits = 16;
constexpr std::uint32_t k_thirty_two_bits = 32;
constexpr std::uint32_t k_sixty_four_bits = std::numeric_limits<std::uint64_t>::digits;
constexpr std::uint32_t k_first_submission_id = 7;
constexpr std::uint32_t k_queue_family = 2;
constexpr std::uint64_t k_first_start_tick = 100;
constexpr std::uint64_t k_second_start_tick = 3'100;
constexpr std::uint64_t k_wrap_start_tick = 65'530;
constexpr std::uint64_t k_later_start_tick = 10;
constexpr std::uint64_t k_control_start_tick = 42;
constexpr std::uint64_t k_wrap_offset = 5;
constexpr double k_short_duration_ns = 1'000.0;
constexpr double k_long_duration_ns = 2'000.0;
constexpr double k_unit_duration_ns = 1.0;
constexpr double k_unit_period_ns = 1.0;
constexpr double k_double_period_ns = 2.0;
constexpr double k_wrap_period_ns = 1'000.0;
constexpr unsigned int k_invalid_utf8_byte = 0xff;
}// namespace

TEST_CASE("Chrome traces retain pass offsets and escape names", "[vkexec][profiling]")
{
  auto const events = std::vector<profile_event>{
    { .name = "first\n\"pass", .submission_id = k_first_submission_id, .queue_family = k_queue_family,
      .region_index = 0,
      .duration_ns = k_long_duration_ns, .status = sample_status::valid, .start_tick = k_first_start_tick,
      .timestamp_valid_bits = k_thirty_two_bits, .timestamp_period_ns = k_unit_period_ns },
    { .name = "second", .submission_id = k_first_submission_id, .queue_family = k_queue_family, .region_index = 1,
      .duration_ns = k_short_duration_ns, .status = sample_status::valid, .start_tick = k_second_start_tick,
      .timestamp_valid_bits = k_thirty_two_bits, .timestamp_period_ns = k_unit_period_ns },
    { .name = "missing", .status = sample_status::unavailable }
  };
  auto output = std::ostringstream{};
  auto const exported = write_chrome_trace(output, events);
  REQUIRE(exported);
  CHECK(exported->omitted_events == 1);
  CHECK(output.str().find("first\\n\\\"pass") != std::string::npos);
  CHECK(output.str().find("\"ts\":3") != std::string::npos);
  CHECK(output.str().find("\"dur\":2") != std::string::npos);
  CHECK(output.str().find("missing") == std::string::npos);
}

TEST_CASE("Chrome traces reject uncalibrated submissions", "[vkexec][profiling]")
{
  auto const events = std::vector<profile_event>{
    { .name = "first", .submission_id = 1, .duration_ns = k_unit_duration_ns, .status = sample_status::valid,
      .timestamp_valid_bits = k_sixty_four_bits, .timestamp_period_ns = k_unit_period_ns },
    { .name = "second", .submission_id = 2, .duration_ns = k_unit_duration_ns, .status = sample_status::valid,
      .timestamp_valid_bits = k_sixty_four_bits, .timestamp_period_ns = k_unit_period_ns }
  };
  auto output = std::ostringstream{};
  CHECK_FALSE(write_chrome_trace(output, events));
  CHECK(output.str().empty());
}

TEST_CASE("Chrome traces handle empty and unavailable collections", "[vkexec][profiling]")
{
  auto output = std::ostringstream{};
  auto const empty = std::vector<profile_event>{};
  auto exported = write_chrome_trace(output, empty);
  REQUIRE(exported);
  CHECK(output.str() == "{\"traceEvents\":[]}");
  CHECK(exported->omitted_events == 0);
  output.str(std::string{});
  auto const unavailable = std::vector<profile_event>{
    { .name = "unavailable", .status = sample_status::unavailable }
  };
  exported = write_chrome_trace(output, unavailable);
  REQUIRE(exported);
  CHECK(exported->omitted_events == 1);
  CHECK(output.str() == "{\"traceEvents\":[]}");
}

TEST_CASE("Chrome traces use recording order across counter wrap", "[vkexec][profiling]")
{
  auto const events = std::vector<profile_event>{
    { .name = "later", .submission_id = 1, .region_index = 1, .duration_ns = k_short_duration_ns,
      .status = sample_status::valid, .start_tick = k_later_start_tick, .timestamp_valid_bits = k_sixteen_bits,
      .timestamp_period_ns = k_wrap_period_ns },
    { .name = "earlier", .submission_id = 1, .region_index = 0, .duration_ns = k_short_duration_ns,
      .status = sample_status::valid, .start_tick = k_wrap_start_tick, .timestamp_valid_bits = k_sixteen_bits,
      .timestamp_period_ns = k_wrap_period_ns }
  };
  auto output = std::ostringstream{};
  REQUIRE(write_chrome_trace(output, events));
  CHECK(output.str().find("\"name\":\"later\"") < output.str().find("\"name\":\"earlier\""));
  CHECK(output.str().find("\"ts\":16") != std::string::npos);
  CHECK(output.str().find("\"ts\":0") != std::string::npos);
}

TEST_CASE("Chrome traces reject incompatible and malformed samples without writing", "[vkexec][profiling]")
{
  auto events = std::vector<profile_event>{
    { .name = "first", .submission_id = 1, .duration_ns = k_unit_duration_ns, .status = sample_status::valid,
      .timestamp_valid_bits = k_sixty_four_bits, .timestamp_period_ns = k_unit_period_ns },
    { .name = "second", .submission_id = 1, .queue_family = 1, .duration_ns = k_unit_duration_ns,
      .status = sample_status::valid,
      .timestamp_valid_bits = k_sixty_four_bits, .timestamp_period_ns = k_unit_period_ns }
  };
  auto output = std::ostringstream{};
  CHECK_FALSE(write_chrome_trace(output, events));
  CHECK(output.str().empty());
  events.at(1).queue_family = 0;
  events.at(1).timestamp_period_ns = k_double_period_ns;
  CHECK_FALSE(write_chrome_trace(output, events));
  events.at(1).timestamp_period_ns = k_unit_period_ns;
  events.at(1).duration_ns = -k_unit_duration_ns;
  CHECK_FALSE(write_chrome_trace(output, events));
  events.at(1).duration_ns = k_unit_duration_ns;
  events.at(1).name = std::string(1, static_cast<char>(k_invalid_utf8_byte));
  CHECK_FALSE(write_chrome_trace(output, events));
  CHECK(output.str().empty());
}

TEST_CASE("Chrome traces report output failures", "[vkexec][profiling]")
{
  auto output = std::ostringstream{};
  output.setstate(std::ios::badbit);
  auto const events = std::vector<profile_event>{};
  CHECK_FALSE(write_chrome_trace(output, events));
}

TEST_CASE("Chrome traces handle a 64-bit timestamp wrap", "[vkexec][profiling]")
{
  auto const events = std::vector<profile_event>{
    { .name = "first", .region_index = 0, .duration_ns = k_unit_duration_ns, .status = sample_status::valid,
      .start_tick = std::numeric_limits<std::uint64_t>::max() - k_wrap_offset,
      .timestamp_valid_bits = k_sixty_four_bits, .timestamp_period_ns = k_wrap_period_ns },
    { .name = "second", .region_index = 1, .duration_ns = k_unit_duration_ns, .status = sample_status::valid,
      .start_tick = k_later_start_tick, .timestamp_valid_bits = k_sixty_four_bits,
      .timestamp_period_ns = k_wrap_period_ns }
  };
  auto output = std::ostringstream{};
  REQUIRE(write_chrome_trace(output, events));
  CHECK(output.str().find("\"ts\":16") != std::string::npos);
}

TEST_CASE("Chrome traces preserve Unicode and escape control characters", "[vkexec][profiling]")
{
  auto const events = std::vector<profile_event>{
    { .name = "GPU μ\t\\\x01", .duration_ns = k_long_duration_ns, .status = sample_status::valid,
      .start_tick = k_control_start_tick, .timestamp_valid_bits = k_thirty_two_bits,
      .timestamp_period_ns = k_unit_period_ns }
  };
  auto output = std::ostringstream{};
  REQUIRE(write_chrome_trace(output, events));
  CHECK(output.str().find("GPU μ\\t\\\\\\u0001") != std::string::npos);
}
