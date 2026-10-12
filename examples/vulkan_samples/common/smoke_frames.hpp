#ifndef VKEXEC_VULKAN_SAMPLES_SMOKE_FRAMES_HPP
#define VKEXEC_VULKAN_SAMPLES_SMOKE_FRAMES_HPP

#include "sync_wait_helpers.hpp"

#include <charconv>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace vkexec::examples {

class smoke_frames
{
public:
  [[nodiscard]] static auto parse(int argc, char const *const *argv) -> smoke_frames
  {
    smoke_frames result;
    constexpr std::string_view k_prefix = "--frames=";
    for (int index = 1; index < argc; ++index) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      std::string_view const argument{ argv[index] };
      if (argument == "--smoke-uncompressed") {
        if (result.uncompressed_) { fail_check("--smoke-uncompressed may only be specified once"); }
        result.uncompressed_ = true;
        continue;
      }
      if (!argument.starts_with(k_prefix)) { continue; }
      if (result.limit_ != 0) { fail_check("--frames may only be specified once"); }
      std::string_view const value = argument.substr(k_prefix.size());
      auto const parsed = std::from_chars(value.data(), value.data() + value.size(), result.limit_);
      if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()
          || result.limit_ == 0) {
        fail_check("--frames requires a positive integer");
      }
    }
    if (result.uncompressed_ && result.limit_ == 0) { fail_check("--smoke-uncompressed requires --frames"); }
    return result;
  }

  [[nodiscard]] static auto is_frame_argument(std::string_view argument) -> bool
  { return argument.starts_with("--frames="); }

  [[nodiscard]] static auto is_smoke_asset_argument(std::string_view argument) -> bool
  { return argument == "--smoke-uncompressed"; }

  static auto reject_other_arguments(int argc, char const *const *argv) -> void
  {
    for (int index = 1; index < argc; ++index) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      if (!is_frame_argument(argv[index])) { fail_check("unknown option"); }
    }
  }

  [[nodiscard]] auto automated() const noexcept -> bool { return limit_ != 0; }
  [[nodiscard]] auto uncompressed() const noexcept -> bool { return uncompressed_; }
  [[nodiscard]] auto should_continue() const noexcept -> bool { return !automated() || scheduled_ < limit_; }
  auto schedule() -> void { ++scheduled_; }
  auto complete() -> void { ++completed_; }
  auto complete_synchronous() -> void
  {
    schedule();
    complete();
  }
  // Call only after the presenter has waited for all scheduled GPU work.
  auto mark_all_drained() -> void
  {
    while (completed_ < scheduled_) { complete(); }
  }
  [[nodiscard]] auto completed() const noexcept -> std::uint64_t { return completed_; }
  auto verify_complete() const -> void
  {
    if (automated() && (scheduled_ != limit_ || completed_ != limit_)) {
      fail_check("window closed before requested frames completed");
    }
    if (automated()) { std::cout << "Completed " << completed_ << " frames\n"; }
  }

private:
  std::uint64_t limit_{ 0 };
  std::uint64_t scheduled_{ 0 };
  std::uint64_t completed_{ 0 };
  bool uncompressed_{ false };
};

}// namespace vkexec::examples

#endif// VKEXEC_VULKAN_SAMPLES_SMOKE_FRAMES_HPP
