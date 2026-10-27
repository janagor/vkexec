#ifndef VKEXEC_PROFILING_MANUAL_CAPTURE_HPP
#define VKEXEC_PROFILING_MANUAL_CAPTURE_HPP

#include <vkexec/context.hpp>
#include <vkexec/result.hpp>
#include <vkexec_profiling/profiler.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vkexec::profiling {

namespace detail {
  class query_allocation;
}

class manual_capture
{
public:
  class region
  {
  public:
    region(region const &) = delete;
    auto operator=(region const &) -> region & = delete;
    region(region &&other) noexcept;
    auto operator=(region &&other) noexcept -> region &;
    ~region();

  private:
    friend class manual_capture;
    region(manual_capture &capture, std::uint32_t index) noexcept : capture_(&capture), index_(index) {}
    manual_capture *capture_;
    std::uint32_t index_;
  };

  [[nodiscard]] static auto create(profiler const &profiler, context &ctx, VkCommandBuffer command_buffer,
                                   queue_ref queue) -> result<manual_capture>;

  manual_capture(manual_capture const &) = delete;
  auto operator=(manual_capture const &) -> manual_capture & = delete;
  manual_capture(manual_capture &&other) noexcept;
  auto operator=(manual_capture &&other) noexcept -> manual_capture &;
  ~manual_capture();

  // Record outside render passes and video coding scopes. Regions may nest.
  [[nodiscard]] auto scoped_region(std::string name) -> result<region>;

  // Call immediately after a successful queue submit, before any throwing work.
  // Keep the fence valid until resolution. Destroying a submitted unresolved
  // capture is a contract violation and terminates the process.
  [[nodiscard]] auto mark_submitted(VkFence fence) -> status;
  // Checks the fence without waiting before reading timestamps. Device loss
  // terminates the capture; other fence errors leave it available to retry.
  [[nodiscard]] auto resolve_after_gpu() -> status;
  // Releases query slots after the submission fence signals, without waiting
  // or collecting samples.
  [[nodiscard]] auto discard_after_gpu() -> status;

private:
  struct sample;
  manual_capture(std::shared_ptr<detail::profiler_state> state, VkCommandBuffer command_buffer, queue_ref queue);
  auto end_region(std::uint32_t index) noexcept -> void;
  auto release() noexcept -> void;
  auto quarantine() noexcept -> void;

  std::shared_ptr<detail::profiler_state> state_;
  VkCommandBuffer command_buffer_{ VK_NULL_HANDLE };
  queue_ref queue_{};
  VkFence fence_{ VK_NULL_HANDLE };
  std::vector<std::unique_ptr<sample>> samples_;
  std::uint32_t active_regions_{ 0 };
  bool submitted_{ false };
  bool resolved_{ false };
  std::uint64_t submission_id_{ 0 };
};

}// namespace vkexec::profiling

#endif// VKEXEC_PROFILING_MANUAL_CAPTURE_HPP
