#ifndef VKEXEC_VMA_COPY_HPP
#define VKEXEC_VMA_COPY_HPP

#include <vkexec/context.hpp>
#include <vkexec/copy.hpp>
#include <vkexec/result.hpp>
#include <vkexec_vma/gpu_buffer.hpp>

#include <cstddef>
#include <span>

namespace vkexec::vma {

[[nodiscard]] auto upload_to_device(context &ctx,
  vma::gpu_buffer &staging,
  vma::gpu_buffer const &device,
  std::span<std::byte const> bytes) -> status;

[[nodiscard]] auto
  download_to_host(context &ctx, vma::gpu_buffer &staging, vma::gpu_buffer const &device, std::span<std::byte> out)
    -> status;

}// namespace vkexec::vma

#endif// VKEXEC_VMA_COPY_HPP
