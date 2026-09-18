#ifndef VKEXEC_DETAIL_COMPUTE_SPECIALIZATION_HPP
#define VKEXEC_DETAIL_COMPUTE_SPECIALIZATION_HPP

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vkexec::detail {

struct uint32_specialization
{
  std::vector<VkSpecializationMapEntry> entries;
  VkSpecializationInfo info{};

  [[nodiscard]] auto get() const noexcept -> VkSpecializationInfo const *
  { return entries.empty() ? nullptr : &info; }
};

[[nodiscard]] inline auto make_uint32_specialization(std::span<std::uint32_t const> values)
  -> uint32_specialization
{
  uint32_specialization specialization{};
  specialization.entries.resize(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    specialization.entries.at(index).constantID = static_cast<std::uint32_t>(index);
    specialization.entries.at(index).offset = static_cast<std::uint32_t>(index * sizeof(std::uint32_t));
    specialization.entries.at(index).size = sizeof(std::uint32_t);
  }
  if (!values.empty()) {
    specialization.info.mapEntryCount = static_cast<std::uint32_t>(specialization.entries.size());
    specialization.info.pMapEntries = specialization.entries.data();
    specialization.info.dataSize = values.size_bytes();
    specialization.info.pData = values.data();
  }
  return specialization;
}

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_COMPUTE_SPECIALIZATION_HPP
