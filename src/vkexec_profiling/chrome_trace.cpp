#include <vkexec_profiling/chrome_trace.hpp>

#include <vkexec/error.hpp>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <string_view>

namespace vkexec::profiling {
namespace {
constexpr double k_nanoseconds_per_microsecond = 1'000.0;
constexpr std::uint32_t k_json_control_limit = 0x20;
constexpr std::uint32_t k_counter_bits = std::numeric_limits<std::uint64_t>::digits;

[[nodiscard]] auto valid_utf8(std::string_view value) -> bool
{
  auto remaining = std::uint32_t{ 0 };
  auto lower = std::uint32_t{ 0x80 };
  auto upper = std::uint32_t{ 0xbf };
  for (auto const character : value) {
    auto const byte = static_cast<unsigned char>(character);
    if (remaining == 0) {
      if (byte <= 0x7f) { continue; }
      if (byte >= 0xc2 && byte <= 0xdf) { remaining = 1; }
      else if (byte == 0xe0) { remaining = 2; lower = 0xa0; }
      else if (byte >= 0xe1 && byte <= 0xec) { remaining = 2; }
      else if (byte == 0xed) { remaining = 2; upper = 0x9f; }
      else if (byte >= 0xee && byte <= 0xef) { remaining = 2; }
      else if (byte == 0xf0) { remaining = 3; lower = 0x90; }
      else if (byte >= 0xf1 && byte <= 0xf3) { remaining = 3; }
      else if (byte == 0xf4) { remaining = 3; upper = 0x8f; }
      else { return false; }
    } else {
      if (byte < lower || byte > upper) { return false; }
      --remaining;
      lower = 0x80;
      upper = 0xbf;
    }
  }
  return remaining == 0;
}

auto write_json_string(std::ostream &output, std::string_view value) -> void
{
  constexpr char k_hex_digits[] = "0123456789abcdef";
  output.put('"');
  for (auto const character : value) {
    auto const byte = static_cast<unsigned char>(character);
    switch (byte) {
    case '"': output << "\\\""; break;
    case '\\': output << "\\\\"; break;
    case '\b': output << "\\b"; break;
    case '\f': output << "\\f"; break;
    case '\n': output << "\\n"; break;
    case '\r': output << "\\r"; break;
    case '\t': output << "\\t"; break;
    default:
      if (byte < k_json_control_limit) {
        output << "\\u00" << k_hex_digits[byte >> 4] << k_hex_digits[byte & 0x0f];
      } else {
        output.put(character);
      }
    }
  }
  output.put('"');
}
}// namespace

auto write_chrome_trace(std::ostream &output, std::span<profile_event const> events) -> result<trace_export_result>
{
  auto result = trace_export_result{};
  profile_event const *origin = nullptr;
  for (auto const &event : events) {
    if (event.status != sample_status::valid) { ++result.omitted_events; continue; }
    if (!valid_utf8(event.name)) { return fail(errc::invalid_argument, "trace event name is not UTF-8"); }
    if (event.timestamp_valid_bits == 0 || event.timestamp_valid_bits > k_counter_bits
        || !std::isfinite(event.timestamp_period_ns) || event.timestamp_period_ns <= 0.0
        || !std::isfinite(event.duration_ns) || event.duration_ns < 0.0) {
      return fail(errc::invalid_argument, "trace contains a valid sample without usable timestamps");
    }
    if (origin == nullptr) { origin = &event; continue; }
    if (event.submission_id != origin->submission_id || event.queue_family != origin->queue_family
        || event.timestamp_valid_bits != origin->timestamp_valid_bits
        || event.timestamp_period_ns != origin->timestamp_period_ns) {
      return fail(errc::unsupported, "trace events do not share one submission timestamp domain");
    }
    if (event.region_index < origin->region_index) { origin = &event; }
  }

  auto json = std::ostringstream{};
  json.imbue(std::locale::classic());
  json << std::setprecision(std::numeric_limits<double>::max_digits10);
  json << "{\"traceEvents\":[";
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
    json << "{\"name\":";
    write_json_string(json, event.name);
    json << ",\"cat\":\"gpu\",\"ph\":\"X\",\"pid\":1,\"tid\":" << event.queue_family
         << ",\"ts\":" << start_us << ",\"dur\":" << duration_us
         << ",\"args\":{\"queue_family\":" << event.queue_family
         << ",\"submission_id\":" << event.submission_id
         << ",\"region_index\":" << event.region_index << "}}";
  }
  json << "]}";
  output << json.str();
  if (!output) { return fail(errc::invalid_argument, "failed to write Chrome trace"); }
  return result;
}

}// namespace vkexec::profiling
