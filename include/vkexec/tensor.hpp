#ifndef VKEXEC_TENSOR_HPP
#define VKEXEC_TENSOR_HPP

//! \file
//! Staging-backed typed tensor using an application-selected buffer allocator.

#include <vkexec/barrier.hpp>
#include <vkexec/buffer.hpp>
#include <vkexec/context.hpp>
#include <vkexec/copy.hpp>
#include <vkexec/detail/normalize_errors.hpp>
#include <vkexec/error.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_allocator.hpp>
#include <vkexec/result.hpp>
#include <vkexec/sender.hpp>

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace vkexec {

template<class A>
concept tensor_buffer_allocator = mapped_buffer_allocator<A> && readback_buffer_resource<typename A::buffer_type>;

namespace detail {
  [[nodiscard]] inline auto
    transfer_tensor_buffers(context &ctx, VkBuffer staging, VkBuffer device, VkDeviceSize bytes, bool upload) -> status
  {
    VKEXEC_TRY_ASSIGN(cmd, ctx.allocate_command_buffer());
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult const begun = vkBeginCommandBuffer(cmd, &begin);
    if (begun != VK_SUCCESS) {
      ctx.free_command_buffer(cmd);
      return fail(begun, "vkBeginCommandBuffer failed for tensor transfer");
    }

    VkBufferMemoryBarrier before{};
    before.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    before.srcAccessMask = upload ? VK_ACCESS_HOST_WRITE_BIT : VK_ACCESS_SHADER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before.buffer = upload ? staging : device;
    before.offset = 0;
    before.size = bytes;
    vkCmdPipelineBarrier(cmd,
      upload ? VK_PIPELINE_STAGE_HOST_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      0,
      0,
      nullptr,
      1,
      &before,
      0,
      nullptr);

    cmd_copy_buffer(cmd, upload ? staging : device, upload ? device : staging, bytes);

    VkBufferMemoryBarrier after{};
    after.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    auto constexpr k_shader_access =
      static_cast<VkAccessFlags>(VK_ACCESS_SHADER_READ_BIT) | static_cast<VkAccessFlags>(VK_ACCESS_SHADER_WRITE_BIT);
    auto constexpr k_host_access = static_cast<VkAccessFlags>(VK_ACCESS_HOST_READ_BIT);
    after.dstAccessMask = upload ? k_shader_access : k_host_access;
    after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    after.buffer = upload ? device : staging;
    after.offset = 0;
    after.size = bytes;
    vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      upload ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_HOST_BIT,
      0,
      0,
      nullptr,
      1,
      &after,
      0,
      nullptr);

    VkResult const ended = vkEndCommandBuffer(cmd);
    if (ended != VK_SUCCESS) {
      ctx.free_command_buffer(cmd);
      return fail(ended, "vkEndCommandBuffer failed for tensor transfer");
    }
    auto submitted = ctx.submit_and_wait(cmd);
    ctx.free_command_buffer(cmd);
    return submitted;
  }
}// namespace detail

template<typename T, readback_buffer_resource B>
  requires std::is_trivially_copyable_v<T>
class tensor
{
public:
  using resource_type = B;

  tensor() = default;
  tensor(tensor const &) = delete;
  auto operator=(tensor const &) -> tensor & = delete;
  tensor(tensor &&) noexcept = default;
  auto operator=(tensor &&) noexcept -> tensor & = default;
  ~tensor() = default;

  [[nodiscard]] auto size() const noexcept -> std::size_t { return host_.size(); }
  [[nodiscard]] auto byte_size() const noexcept -> VkDeviceSize { return size() * sizeof(T); }
  [[nodiscard]] auto vk_buffer() const noexcept -> VkBuffer { return gpu_ ? gpu_->device.handle() : VK_NULL_HANDLE; }
  [[nodiscard]] auto data() noexcept -> T * { return host_.data(); }
  [[nodiscard]] auto data() const noexcept -> T const * { return host_.data(); }
  [[nodiscard]] auto span() noexcept -> std::span<T> { return host_; }
  [[nodiscard]] auto span() const noexcept -> std::span<T const> { return host_; }
  [[nodiscard]] auto staging() noexcept -> resource_type & { return gpu_->staging; }
  [[nodiscard]] auto staging() const noexcept -> resource_type const & { return gpu_->staging; }
  [[nodiscard]] auto device() noexcept -> resource_type & { return gpu_->device; }
  [[nodiscard]] auto device() const noexcept -> resource_type const & { return gpu_->device; }

  [[nodiscard]] auto storage_binding(std::uint32_t binding) const noexcept -> vkexec::storage_binding
  { return { .buffer = vk_buffer(), .byte_size = byte_size(), .binding = binding }; }

  [[nodiscard]] auto upload(context &ctx) -> status
  {
    if (!gpu_) { return fail(errc::invalid_argument, "upload requires a non-empty tensor"); }
    if (staging().size() < byte_size() || device().size() < byte_size()) {
      return fail(errc::out_of_range, "tensor allocation is too small");
    }
    auto mapped = staging().mapped();
    if (mapped.size() < byte_size()) { return fail(errc::out_of_range, "tensor staging map is too small"); }
    std::memcpy(mapped.data(), data(), static_cast<std::size_t>(byte_size()));
    if (auto flushed = staging().flush(); !flushed) { return flushed; }
    return detail::transfer_tensor_buffers(ctx, staging().handle(), device().handle(), byte_size(), true);
  }

  [[nodiscard]] auto download(context &ctx) -> status
  {
    if (!gpu_) { return fail(errc::invalid_argument, "download requires a non-empty tensor"); }
    if (staging().size() < byte_size() || device().size() < byte_size()) {
      return fail(errc::out_of_range, "tensor allocation is too small");
    }
    auto mapped = staging().mapped();
    if (mapped.size() < byte_size()) { return fail(errc::out_of_range, "tensor staging map is too small"); }
    VKEXEC_TRY(detail::transfer_tensor_buffers(ctx, staging().handle(), device().handle(), byte_size(), false));
    if (auto invalidated = staging().invalidate(); !invalidated) { return invalidated; }
    std::memcpy(data(), mapped.data(), static_cast<std::size_t>(byte_size()));
    return {};
  }

  [[nodiscard]] static auto
    make_initialized(resource_type staging_buffer, resource_type device_buffer, std::vector<T> host) -> result<tensor>
  {
    auto const bytes = static_cast<VkDeviceSize>(host.size()) * sizeof(T);
    auto mapped = staging_buffer.mapped();
    if (staging_buffer.size() < bytes || device_buffer.size() < bytes || mapped.size() < bytes) {
      return fail(errc::unsupported, "tensor allocation is too small");
    }
    std::memcpy(mapped.data(), host.data(), static_cast<std::size_t>(bytes));
    if (auto flushed = staging_buffer.flush(); !flushed) { return fail(flushed.error()); }
    auto gpu = std::make_unique<gpu_storage>(gpu_storage{ std::move(staging_buffer), std::move(device_buffer) });
    return tensor{ std::move(host), std::move(gpu) };
  }

private:
  struct gpu_storage
  {
    resource_type staging;
    resource_type device;
  };

  explicit tensor(std::vector<T> host, std::unique_ptr<gpu_storage> gpu) noexcept
    : host_(std::move(host)), gpu_(std::move(gpu))
  {}

  std::vector<T> host_;
  std::unique_ptr<gpu_storage> gpu_;
};

namespace factory {
  namespace detail {
    template<typename T, tensor_buffer_allocator A, class PreparedSender>
      requires std::is_trivially_copyable_v<T>
    [[nodiscard]] auto make_tensor_from_values_sender(A &allocator, PreparedSender prepared)
    {
      using resource_type = A::buffer_type;
      auto composed = std::move(prepared) | stdexec::let_value([&allocator](std::vector<T> &host) -> auto {
        auto const bytes = static_cast<VkDeviceSize>(host.size()) * sizeof(T);
        return allocate_buffer(allocator,
                 buffer_create_info{
                   .size = bytes,
                   .usage = static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
                            | static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_TRANSFER_DST_BIT),
                   .memory = memory_domain::staging,
                 })
               | stdexec::let_value([&allocator, &host, bytes](resource_type &staging) -> auto {
                   return allocate_buffer(allocator,
                            buffer_create_info{
                              .size = bytes,
                              .usage = static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
                                       | static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
                                       | static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_TRANSFER_DST_BIT),
                              .memory = memory_domain::device_local,
                              .shader_device_address = true,
                            })
                          | stdexec::let_value([&host, &staging](resource_type &device) -> auto {
                              return make_sender([&host, &staging, &device]() -> result<tensor<T, resource_type>> {
                                return tensor<T, resource_type>::make_initialized(
                                  std::move(staging), std::move(device), std::move(host));
                              });
                            });
                 });
      });
      return ::vkexec::detail::normalize_errors(std::move(composed));
    }
  }// namespace detail

  struct make_tensor_t
  {
    template<typename T, tensor_buffer_allocator A>
      requires std::is_trivially_copyable_v<T>
    [[nodiscard]] auto operator()(A &allocator, std::size_t count, T fill) const
    {
      auto prepared = make_sender([count, fill = std::move(fill)]() mutable -> result<std::vector<T>> {
        if (count == 0 || count > std::numeric_limits<VkDeviceSize>::max() / sizeof(T)) {
          return fail(errc::invalid_argument, "tensor element count is invalid");
        }
        return std::vector<T>(count, fill);
      });
      return detail::make_tensor_from_values_sender<T>(allocator, std::move(prepared));
    }

    template<typename T, tensor_buffer_allocator A>
      requires std::is_trivially_copyable_v<T>
    [[nodiscard]] auto operator()(A &allocator, std::span<T const> values) const
    {
      auto prepared = make_sender([values]() -> result<std::vector<T>> {
        if (values.empty() || values.size() > std::numeric_limits<VkDeviceSize>::max() / sizeof(T)) {
          return fail(errc::invalid_argument, "tensor element count is invalid");
        }
        return std::vector<T>(values.begin(), values.end());
      });
      return detail::make_tensor_from_values_sender<T>(allocator, std::move(prepared));
    }

    template<typename T, tensor_buffer_allocator A>
      requires std::is_trivially_copyable_v<T>
    [[nodiscard]] auto operator()(A &allocator, std::vector<T> values) const
    {
      auto prepared = make_sender([values = std::move(values)]() mutable -> result<std::vector<T>> {
        if (values.empty() || values.size() > std::numeric_limits<VkDeviceSize>::max() / sizeof(T)) {
          return fail(errc::invalid_argument, "tensor element count is invalid");
        }
        return std::move(values);
      });
      return detail::make_tensor_from_values_sender<T>(allocator, std::move(prepared));
    }
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  inline constexpr make_tensor_t make_tensor{};
}// namespace factory

}// namespace vkexec

#endif// VKEXEC_TENSOR_HPP
