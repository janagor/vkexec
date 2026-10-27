#include <vkexec_profiling/chrome_trace.hpp>

#include <vkexec/error.hpp>
#include <vkexec/result.hpp>
#include <vkexec_profiling/profile_event.hpp>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <span>
#include <sstream>
#include <string_view>

namespace vkexec::profiling {
namespace {
  constexpr double k_nanoseconds_per_microsecond = 1'000.0;
  constexpr std::uint32_t k_json_control_limit = 0x20;
  constexpr std::uint32_t k_counter_bits = std::numeric_limits<std::uint64_t>::digits;
  constexpr std::uint32_t k_ascii_limit = 0x7f;
  constexpr std::uint32_t k_continuation_min = 0x80;
  constexpr std::uint32_t k_continuation_max = 0xbf;
  constexpr std::uint32_t k_two_byte_min = 0xc2;
  constexpr std::uint32_t k_two_byte_max = 0xdf;
  constexpr std::uint32_t k_three_byte_first = 0xe0;
  constexpr std::uint32_t k_three_byte_first_min = 0xa0;
  constexpr std::uint32_t k_three_byte_middle_min = 0xe1;
  constexpr std::uint32_t k_three_byte_middle_max = 0xec;
  constexpr std::uint32_t k_three_byte_surrogate = 0xed;
  constexpr std::uint32_t k_three_byte_surrogate_max = 0x9f;
  constexpr std::uint32_t k_three_byte_last_min = 0xee;
  constexpr std::uint32_t k_three_byte_last_max = 0xef;
  constexpr std::uint32_t k_four_byte_first = 0xf0;
  constexpr std::uint32_t k_four_byte_first_min = 0x90;
  constexpr std::uint32_t k_four_byte_middle_min = 0xf1;
  constexpr std::uint32_t k_four_byte_middle_max = 0xf3;
  constexpr std::uint32_t k_four_byte_last = 0xf4;
  constexpr std::uint32_t k_four_byte_last_max = 0x8f;
  constexpr std::uint32_t k_hex_radix = 16;
  constexpr std::string_view k_hex_digits = "0123456789abcdef";

  [[nodiscard]] auto
    classify_utf8_lead(std::uint32_t byte, std::uint32_t &remaining, std::uint32_t &lower, std::uint32_t &upper) -> bool
  {
    if (byte >= k_two_byte_min && byte <= k_two_byte_max) {
      remaining = 1;
    } else if (byte == k_three_byte_first) {
      remaining = 2;
      lower = k_three_byte_first_min;
    } else if ((byte >= k_three_byte_middle_min && byte <= k_three_byte_middle_max)
               || (byte >= k_three_byte_last_min && byte <= k_three_byte_last_max)) {
      remaining = 2;
    } else if (byte == k_three_byte_surrogate) {
      remaining = 2;
      upper = k_three_byte_surrogate_max;
    } else if (byte == k_four_byte_first) {
      remaining = 3;
      lower = k_four_byte_first_min;
    } else if (byte >= k_four_byte_middle_min && byte <= k_four_byte_middle_max) {
      remaining = 3;
    } else if (byte == k_four_byte_last) {
      remaining = 3;
      upper = k_four_byte_last_max;
    } else {
      return false;
    }
    return true;
  }

  [[nodiscard]] auto valid_utf8(std::string_view value) -> bool
  {
    auto remaining = std::uint32_t{ 0 };
    auto lower = k_continuation_min;
    auto upper = k_continuation_max;
    for (auto const character : value) {
      auto const byte = static_cast<unsigned char>(character);
      if (remaining == 0) {
        if (byte <= k_ascii_limit) { continue; }
        if (!classify_utf8_lead(byte, remaining, lower, upper)) { return false; }
      } else {
        if (byte < lower || byte > upper) { return false; }
        --remaining;
        lower = k_continuation_min;
        upper = k_continuation_max;
      }
    }
    return remaining == 0;
  }

  auto write_json_string(std::ostream &output, std::string_view value) -> void
  {
    output.put('"');
    for (auto const character : value) {
      auto const byte = static_cast<unsigned char>(character);
      switch (byte) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (byte < k_json_control_limit) {
          output << "\\u00" << k_hex_digits.at(byte / k_hex_radix) << k_hex_digits.at(byte % k_hex_radix);
        } else {
          output.put(character);
        }
      }
    }
    output.put('"');
  }

  [[nodiscard]] auto has_usable_timestamps(profile_event const &event) -> bool
  {
    return event.timestamp_valid_bits != 0 && event.timestamp_valid_bits <= k_counter_bits
           && std::isfinite(event.timestamp_period_ns) && event.timestamp_period_ns > 0.0
           && std::isfinite(event.duration_ns) && event.duration_ns >= 0.0;
  }

  [[nodiscard]] auto shares_timestamp_domain(profile_event const &event, profile_event const &origin) -> bool
  {
    return event.submission_id == origin.submission_id && event.queue_family == origin.queue_family
           && event.timestamp_valid_bits == origin.timestamp_valid_bits
           && event.timestamp_period_ns == origin.timestamp_period_ns;
  }
}// namespace

auto write_chrome_trace(std::ostream &output, std::span<profile_event const> events) -> result<trace_export_result>
{
  auto result = trace_export_result{};
  profile_event const *origin = nullptr;
  for (auto const &event : events) {
    if (event.status != sample_status::valid) {
      ++result.omitted_events;
      continue;
    }
    if (!valid_utf8(event.name)) { return fail(errc::invalid_argument, "trace event name is not UTF-8"); }
    if (!has_usable_timestamps(event)) {
      return fail(errc::invalid_argument, "trace contains a valid sample without usable timestamps");
    }
    if (origin == nullptr) {
      origin = &event;
      continue;
    }
    if (!shares_timestamp_domain(event, *origin)) {
      return fail(errc::unsupported, "trace events do not share one submission timestamp domain");
    }
    if (event.region_index < origin->region_index) { origin = &event; }
  }

  auto json = std::ostringstream{};
  json.imbue(std::locale::classic());
  json << std::setprecision(std::numeric_limits<double>::max_digits10);
  json << R"({"traceEvents":[)";
  auto first = true;
  for (auto const &event : events) {
    if (event.status != sample_status::valid) { continue; }
    auto ticks = event.start_tick - origin->start_tick;
    if (event.timestamp_valid_bits < k_counter_bits) {
      ticks &= (std::uint64_t{ 1 } << event.timestamp_valid_bits) - 1;
    }
    auto const start_us = static_cast<double>(ticks) * event.timestamp_period_ns / k_nanoseconds_per_microsecond;
    auto const duration_us = event.duration_ns / k_nanoseconds_per_microsecond;
    if (!std::isfinite(start_us) || !std::isfinite(duration_us)) {
      return fail(errc::invalid_argument, "trace timestamp conversion overflowed");
    }
    if (!first) { json << ','; }
    first = false;
    json << R"({"name":)";
    write_json_string(json, event.name);
    json << R"(,"cat":"gpu","ph":"X","pid":1,"tid":)" << event.queue_family << R"(,"ts":)" << start_us << R"(,"dur":)"
         << duration_us << R"(,"args":{"queue_family":)" << event.queue_family << R"(,"submission_id":)"
         << event.submission_id << R"(,"region_index":)" << event.region_index << "}}";
  }
  json << "]}";
  output << json.str();
  if (!output) { return fail(errc::invalid_argument, "failed to write Chrome trace"); }
  return result;
}

}// namespace vkexec::profiling
