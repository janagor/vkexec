#include <catch2/catch_test_macros.hpp>

#include <vkexec_profiling/chrome_trace.hpp>

#include <limits>
#include <sstream>
#include <string>
#include <vector>

TEST_CASE("Chrome traces retain pass offsets and escape names", "[vkexec][profiling]")
{
  using namespace vkexec::profiling;
  auto const events = std::vector<profile_event>{
    { .name = "first\n\"pass", .submission_id = 7, .queue_family = 2, .region_index = 0,
      .duration_ns = 2'000.0, .status = sample_status::valid, .start_tick = 100,
      .timestamp_valid_bits = 32, .timestamp_period_ns = 1.0 },
    { .name = "second", .submission_id = 7, .queue_family = 2, .region_index = 1,
      .duration_ns = 1'000.0, .status = sample_status::valid, .start_tick = 3'100,
      .timestamp_valid_bits = 32, .timestamp_period_ns = 1.0 },
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
  using namespace vkexec::profiling;
  auto const events = std::vector<profile_event>{
    { .name = "first", .submission_id = 1, .duration_ns = 1.0, .status = sample_status::valid,
      .timestamp_valid_bits = 64, .timestamp_period_ns = 1.0 },
    { .name = "second", .submission_id = 2, .duration_ns = 1.0, .status = sample_status::valid,
      .timestamp_valid_bits = 64, .timestamp_period_ns = 1.0 }
  };
  auto output = std::ostringstream{};
  CHECK_FALSE(write_chrome_trace(output, events));
  CHECK(output.str().empty());
}

TEST_CASE("Chrome traces handle empty and unavailable collections", "[vkexec][profiling]")
{
  using namespace vkexec::profiling;
  auto output = std::ostringstream{};
  auto const empty = std::vector<profile_event>{};
  auto exported = write_chrome_trace(output, empty);
  REQUIRE(exported);
  CHECK(output.str() == "{\"traceEvents\":[]}");
  CHECK(exported->omitted_events == 0);
  output.str(std::string{});
  auto const unavailable = std::vector<profile_event>{ { .name = "unavailable", .status = sample_status::unavailable } };
  exported = write_chrome_trace(output, unavailable);
  REQUIRE(exported);
  CHECK(exported->omitted_events == 1);
  CHECK(output.str() == "{\"traceEvents\":[]}");
}

TEST_CASE("Chrome traces use recording order across counter wrap", "[vkexec][profiling]")
{
  using namespace vkexec::profiling;
  auto const events = std::vector<profile_event>{
    { .name = "later", .submission_id = 1, .region_index = 1, .duration_ns = 1'000.0,
      .status = sample_status::valid, .start_tick = 10, .timestamp_valid_bits = 16,
      .timestamp_period_ns = 1'000.0 },
    { .name = "earlier", .submission_id = 1, .region_index = 0, .duration_ns = 1'000.0,
      .status = sample_status::valid, .start_tick = 65'530, .timestamp_valid_bits = 16,
      .timestamp_period_ns = 1'000.0 }
  };
  auto output = std::ostringstream{};
  REQUIRE(write_chrome_trace(output, events));
  CHECK(output.str().find("\"name\":\"later\"") < output.str().find("\"name\":\"earlier\""));
  CHECK(output.str().find("\"ts\":16") != std::string::npos);
  CHECK(output.str().find("\"ts\":0") != std::string::npos);
}

TEST_CASE("Chrome traces reject incompatible and malformed samples without writing", "[vkexec][profiling]")
{
  using namespace vkexec::profiling;
  auto events = std::vector<profile_event>{
    { .name = "first", .submission_id = 1, .duration_ns = 1.0, .status = sample_status::valid,
      .timestamp_valid_bits = 64, .timestamp_period_ns = 1.0 },
    { .name = "second", .submission_id = 1, .queue_family = 1, .duration_ns = 1.0, .status = sample_status::valid,
      .timestamp_valid_bits = 64, .timestamp_period_ns = 1.0 }
  };
  auto output = std::ostringstream{};
  CHECK_FALSE(write_chrome_trace(output, events));
  CHECK(output.str().empty());
  events[1].queue_family = 0;
  events[1].timestamp_period_ns = 2.0;
  CHECK_FALSE(write_chrome_trace(output, events));
  events[1].timestamp_period_ns = 1.0;
  events[1].duration_ns = -1.0;
  CHECK_FALSE(write_chrome_trace(output, events));
  events[1].duration_ns = 1.0;
  events[1].name = std::string(1, static_cast<char>(0xff));
  CHECK_FALSE(write_chrome_trace(output, events));
  CHECK(output.str().empty());
}

TEST_CASE("Chrome traces report output failures", "[vkexec][profiling]")
{
  using namespace vkexec::profiling;
  auto output = std::ostringstream{};
  output.setstate(std::ios::badbit);
  auto const events = std::vector<profile_event>{};
  CHECK_FALSE(write_chrome_trace(output, events));
}

TEST_CASE("Chrome traces handle a 64-bit timestamp wrap", "[vkexec][profiling]")
{
  using namespace vkexec::profiling;
  auto const events = std::vector<profile_event>{
    { .name = "first", .region_index = 0, .duration_ns = 1.0, .status = sample_status::valid,
      .start_tick = std::numeric_limits<std::uint64_t>::max() - 5,
      .timestamp_valid_bits = 64, .timestamp_period_ns = 1'000.0 },
    { .name = "second", .region_index = 1, .duration_ns = 1.0, .status = sample_status::valid,
      .start_tick = 10, .timestamp_valid_bits = 64, .timestamp_period_ns = 1'000.0 }
  };
  auto output = std::ostringstream{};
  REQUIRE(write_chrome_trace(output, events));
  CHECK(output.str().find("\"ts\":16") != std::string::npos);
}

TEST_CASE("Chrome traces preserve Unicode and escape control characters", "[vkexec][profiling]")
{
  using namespace vkexec::profiling;
  auto const events = std::vector<profile_event>{
    { .name = "GPU μ\t\\\x01", .duration_ns = 2'000.0, .status = sample_status::valid,
      .start_tick = 42, .timestamp_valid_bits = 32, .timestamp_period_ns = 1.0 }
  };
  auto output = std::ostringstream{};
  REQUIRE(write_chrome_trace(output, events));
  CHECK(output.str().find("GPU μ\\t\\\\\\u0001") != std::string::npos);
}
