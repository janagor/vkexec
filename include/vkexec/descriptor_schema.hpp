#ifndef VKEXEC_DESCRIPTOR_SCHEMA_HPP
#define VKEXEC_DESCRIPTOR_SCHEMA_HPP

//! \file
//! Compile-time logical storage-buffer binding schemas.

#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace vkexec {

//! One logical storage-buffer slot in a descriptor schema.
template<std::uint32_t Slot, buffer_access Access = buffer_access::readwrite>
struct storage_buffer
{
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr std::uint32_t slot = Slot;
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr buffer_access access = Access;
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr resource_kind kind = resource_kind::storage_buffer;
};

//! One logical storage-image slot in a descriptor schema.
template<std::uint32_t Slot>
struct storage_image
{
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr std::uint32_t slot = Slot;
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr buffer_access access = buffer_access::readwrite;
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr resource_kind kind = resource_kind::storage_image;
};

//! One logical sampled-image slot in a descriptor schema.
template<std::uint32_t Slot>
struct sampled_image
{
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr std::uint32_t slot = Slot;
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr buffer_access access = buffer_access::readonly;
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr resource_kind kind = resource_kind::sampled_image;
};

//! One logical sampler slot; named to avoid colliding with the owning `sampler` type.
template<std::uint32_t Slot>
struct sampler_binding
{
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr std::uint32_t slot = Slot;
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr buffer_access access = buffer_access::readonly;
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr resource_kind kind = resource_kind::sampler;
};

namespace detail {

  template<class Entry>
  concept descriptor_schema_entry = requires {
    { Entry::slot } -> std::convertible_to<std::uint32_t>;
    { Entry::access } -> std::convertible_to<buffer_access>;
    { Entry::kind } -> std::convertible_to<resource_kind>;
  };

  template<class... Entries>
  consteval auto descriptor_schema_slots_unique() -> bool
  {
    constexpr std::array<std::uint32_t, sizeof...(Entries)> k_slots{ Entries::slot... };
    for (std::size_t current = 0; current < k_slots.size(); ++current) {
      for (std::size_t other = current + 1; other < k_slots.size(); ++other) {
        if (k_slots.at(current) == k_slots.at(other)) { return false; }
      }
    }
    return true;
  }

  template<class... Entries>
  consteval auto descriptor_schema_slots_sorted() -> bool
  {
    constexpr std::array<std::uint32_t, sizeof...(Entries)> k_slots{ Entries::slot... };
    for (std::size_t index = 1; index < k_slots.size(); ++index) {
      if (k_slots.at(index - 1) > k_slots.at(index)) { return false; }
    }
    return true;
  }

  template<class Entry, class Resource>
  [[nodiscard]] auto schema_resource(Resource &&resource) -> resource_ref
  {
    auto result = static_cast<resource_ref>(std::forward<Resource>(resource));
    result.kind = Entry::kind;
    return result;
  }

}// namespace detail

//! Ordered compile-time descriptor contract for logical resource slots.
template<class... Entries>
struct descriptor_schema
{
  static_assert((detail::descriptor_schema_entry<Entries> && ...), "descriptor_schema entries must be resource slots");
  static_assert(detail::descriptor_schema_slots_unique<Entries...>(), "descriptor_schema slots must be unique");
  static_assert(detail::descriptor_schema_slots_sorted<Entries...>(), "descriptor_schema slots must be sorted");

  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr std::size_t binding_count = sizeof...(Entries);
};

//! Derives a classic compute layout from a schema's slots and access modes.
template<class... Entries>
[[nodiscard]] auto layout_desc_from_schema(descriptor_schema<Entries...> /*schema*/,
  std::size_t push_constant_size = 0,
  std::array<std::uint32_t, 3> local_size = k_default_local_size) -> layout_desc
{
  return layout_desc{ .binding_kinds = { Entries::kind... },
    .binding_slots = { Entries::slot... },
    .bindings = { Entries::access... },
    .push_constant_size = push_constant_size,
    .specialization = {},
    .local_size = local_size };
}

//! Builds a resource table whose logical slots come from `schema`.
template<class... Entries, class... Resources>
  requires(sizeof...(Entries) == sizeof...(Resources))
       && (std::convertible_to<Resources &&, resource_ref> && ...)
[[nodiscard]] auto make_resource_table(descriptor_schema<Entries...> /*schema*/, Resources &&...resources)
  -> resource_table
{
  std::vector<resource_binding> entries;
  entries.reserve(sizeof...(Entries));
  (entries.push_back(resource_binding{
     .slot = Entries::slot, .resource = detail::schema_resource<Entries>(std::forward<Resources>(resources)) }),
    ...);
  return resource_table{ std::move(entries) };
}

}// namespace vkexec

#endif// VKEXEC_DESCRIPTOR_SCHEMA_HPP
