#include <array>
#include <catch2/catch_test_macros.hpp>
#include <map>
#include <tuple>
#include <vkexec/detail/resource_barrier_coalescing.hpp>

using namespace vkexec;
using namespace vkexec::detail;

namespace {
template<class B> void require_equal(B const &a, B const &b)
{
  REQUIRE(same_barrier_scope(a, b));
  if constexpr (requires { a.buffer; }) {
    REQUIRE(a.buffer == b.buffer);
    REQUIRE(a.offset == b.offset);
    REQUIRE(a.size == b.size);
  } else {
    REQUIRE(a.image == b.image);
    REQUIRE(a.old_layout == b.old_layout);
    REQUIRE(a.new_layout == b.new_layout);
    REQUIRE(a.range.aspectMask == b.range.aspectMask);
    REQUIRE(a.range.baseMipLevel == b.range.baseMipLevel);
    REQUIRE(a.range.levelCount == b.range.levelCount);
    REQUIRE(a.range.baseArrayLayer == b.range.baseArrayLayer);
    REQUIRE(a.range.layerCount == b.range.layerCount);
  }
}
template<class B> void scope_fields(B base)
{
  REQUIRE(same_barrier_scope(base, base));
  auto check = [&](auto member) {
    auto changed = base;
    ++(changed.*member);
    REQUIRE_FALSE(same_barrier_scope(base, changed));
  };
  check(&B::src_stage);
  check(&B::dst_stage);
  check(&B::src_access);
  check(&B::dst_access);
  check(&B::src_queue_family);
  check(&B::dst_queue_family);
}
using Metadata = std::tuple<VkPipelineStageFlags2,
  VkPipelineStageFlags2,
  VkAccessFlags2,
  VkAccessFlags2,
  unsigned,
  unsigned,
  VkImageLayout,
  VkImageLayout>;
template<class B> auto metadata(B const &b)
{
  VkImageLayout old{}, next{};
  if constexpr (requires { b.image; }) {
    old = b.old_layout;
    next = b.new_layout;
  }
  return Metadata{
    b.src_stage, b.dst_stage, b.src_access, b.dst_access, b.src_queue_family, b.dst_queue_family, old, next
  };
}
auto buffer_cells(std::vector<buffer_barrier_params> const &barriers)
{
  std::map<std::pair<VkBuffer, unsigned>, Metadata> cells;
  for (auto const &b : barriers)
    for (unsigned byte = static_cast<unsigned>(b.offset); byte < b.offset + b.size; ++byte) {
      REQUIRE(cells.emplace(std::pair{ b.buffer, byte }, metadata(b)).second);
    }
  return cells;
}
auto image_cells(std::vector<image_barrier_params> const &barriers)
{
  std::map<std::tuple<VkImage, unsigned, unsigned, unsigned>, Metadata> cells;
  for (auto const &b : barriers)
    for (unsigned aspect = 1; aspect <= VK_IMAGE_ASPECT_STENCIL_BIT; aspect <<= 1U)
      if ((b.range.aspectMask & aspect) != 0)
        for (unsigned mip = b.range.baseMipLevel; mip < b.range.baseMipLevel + b.range.levelCount; ++mip)
          for (unsigned layer = b.range.baseArrayLayer; layer < b.range.baseArrayLayer + b.range.layerCount; ++layer) {
            REQUIRE(cells.emplace(std::tuple{ b.image, aspect, mip, layer }, metadata(b)).second);
          }
  return cells;
}
}// namespace

TEST_CASE("each synchronization field participates in barrier scope equality", "[vkexec][barrier-coalescing]")
{
  scope_fields(buffer_barrier_params{});
  scope_fields(image_barrier_params{});
}

TEST_CASE(
  "buffer barrier adjacency preserves exact ranges and rejects "
  "incompatible metadata without mutation",
  "[vkexec][barrier-coalescing]")
{
  int storage{}, other{};
  buffer_barrier_params base{ .buffer = static_cast<VkBuffer>(static_cast<void *>(&storage)),
    .offset = 8,
    .size = 8,
    .src_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
    .dst_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
    .dst_access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT };
  auto adjacent = base;
  adjacent.offset = 16;
  auto merged = base;
  REQUIRE(merge_barriers(merged, adjacent));
  REQUIRE(merged.offset == 8);
  REQUIRE(merged.size == 16);
  REQUIRE(same_barrier_scope(merged, base));
  merged = adjacent;
  REQUIRE(merge_barriers(merged, base));
  REQUIRE(merged.offset == 8);
  REQUIRE(merged.size == 16);
  auto reject = [&](auto lhs, auto rhs) {
    auto original = lhs;
    REQUIRE_FALSE(merge_barriers(lhs, rhs));
    require_equal(lhs, original);
  };
  for (auto offset : { 12U, 24U }) {
    auto rhs = adjacent;
    rhs.offset = offset;
    reject(base, rhs);
  }
  auto rhs = adjacent;
  rhs.buffer = static_cast<VkBuffer>(static_cast<void *>(&other));
  reject(base, rhs);
  auto change = [&](auto member) {
    auto right = adjacent;
    ++(right.*member);
    reject(base, right);
  };
  change(&buffer_barrier_params::src_stage);
  change(&buffer_barrier_params::dst_stage);
  change(&buffer_barrier_params::src_access);
  change(&buffer_barrier_params::dst_access);
  change(&buffer_barrier_params::src_queue_family);
  change(&buffer_barrier_params::dst_queue_family);
  auto transfer = base;
  transfer.src_queue_family = 1;
  transfer.dst_queue_family = 2;
  rhs = transfer;
  rhs.offset = 16;
  reject(transfer, rhs);
  rhs = adjacent;
  rhs.size = 0;
  reject(base, rhs);
  auto invalid = base;
  invalid.size = 0;
  reject(invalid, adjacent);
  rhs = adjacent;
  rhs.offset = ~VkDeviceSize{};
  reject(base, rhs);
}

TEST_CASE("image barrier merges require rectangular geometry and identical semantics", "[vkexec][barrier-coalescing]")
{
  int storage{}, other{};
  image_barrier_params base{ .image = static_cast<VkImage>(static_cast<void *>(&storage)),
    .old_layout = VK_IMAGE_LAYOUT_GENERAL,
    .new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
  for (bool mip : { false, true }) {
    auto rhs = base;
    if (mip)
      rhs.range.baseMipLevel = 1;
    else
      rhs.range.baseArrayLayer = 1;
    auto lhs = base;
    REQUIRE(merge_barriers(lhs, rhs));
    REQUIRE(lhs.range.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT);
    REQUIRE(lhs.range.baseMipLevel == 0);
    REQUIRE(lhs.range.baseArrayLayer == 0);
    REQUIRE(lhs.range.levelCount == (mip ? 2 : 1));
    REQUIRE(lhs.range.layerCount == (mip ? 1 : 2));
    require_equal(rhs, rhs);
  }
  auto adjacent = base;
  adjacent.range.baseMipLevel = 1;
  auto reject = [&](auto lhs, auto rhs) {
    auto original = lhs;
    REQUIRE_FALSE(merge_barriers(lhs, rhs));
    require_equal(lhs, original);
  };
  auto rhs = adjacent;
  rhs.range.baseArrayLayer = 1;
  reject(base, rhs);
  rhs = adjacent;
  rhs.range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  reject(base, rhs);
  rhs = adjacent;
  rhs.old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  reject(base, rhs);
  rhs = adjacent;
  rhs.new_layout = VK_IMAGE_LAYOUT_GENERAL;
  reject(base, rhs);
  rhs = adjacent;
  rhs.image = static_cast<VkImage>(static_cast<void *>(&other));
  reject(base, rhs);
  auto change = [&](auto member) {
    auto right = adjacent;
    ++(right.*member);
    reject(base, right);
  };
  change(&image_barrier_params::src_stage);
  change(&image_barrier_params::dst_stage);
  change(&image_barrier_params::src_access);
  change(&image_barrier_params::dst_access);
  change(&image_barrier_params::src_queue_family);
  change(&image_barrier_params::dst_queue_family);
  auto transfer = base;
  transfer.src_queue_family = 1;
  transfer.dst_queue_family = 2;
  rhs = transfer;
  rhs.range.baseMipLevel = 1;
  reject(transfer, rhs);
  rhs = adjacent;
  rhs.range.levelCount = 0;
  reject(base, rhs);
  auto invalid = base;
  invalid.range.layerCount = 0;
  reject(invalid, adjacent);
}

TEST_CASE(
  "coalescing reaches a stable fixed point and preserves byte and cell "
  "reference models",
  "[vkexec][barrier-coalescing]")
{
  std::vector<buffer_barrier_params> buffers;
  buffers = { { .offset = 4, .size = 4 } };
  auto single = buffers;
  coalesce_barriers(buffers);
  require_equal(buffers[0], single[0]);
  for (auto offsets : { std::array{ 0U, 4U }, std::array{ 4U, 0U } }) {
    buffers.clear();
    for (auto offset : offsets) buffers.push_back({ .offset = offset, .size = 4 });
    coalesce_barriers(buffers);
    REQUIRE(buffers.size() == 1);
    REQUIRE(buffers[0].offset == 0);
    REQUIRE(buffers[0].size == 8);
  }
  std::vector<buffer_barrier_params> empty;
  coalesce_barriers(empty);
  REQUIRE(empty.empty());
  buffers = { { .offset = 0, .size = 4 },
    { .offset = 12, .size = 4, .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT },
    { .offset = 8, .size = 4 },
    { .offset = 16, .size = 4, .src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT },
    { .offset = 4, .size = 4 },
    { .offset = 20, .size = 4, .src_queue_family = 1, .dst_queue_family = 2 },
    { .offset = 24, .size = 4, .src_queue_family = 1, .dst_queue_family = 2 } };
  auto bytes = buffer_cells(buffers);
  coalesce_barriers(buffers);
  REQUIRE(buffers.size() == 4);
  REQUIRE(buffer_cells(buffers) == bytes);
  auto once = buffers;
  coalesce_barriers(buffers);
  REQUIRE(buffers.size() == once.size());
  for (unsigned i = 0; i < buffers.size(); ++i) require_equal(buffers[i], once[i]);
  std::vector<image_barrier_params> images;
  coalesce_barriers(images);
  REQUIRE(images.empty());
  // Deliberately leave the bridging row until last: merging must rescan earlier
  // pairs.
  for (unsigned mip : { 0U, 2U, 3U, 1U })
    for (unsigned layer : { 0U, 2U, 3U, 1U }) {
      image_barrier_params b;
      b.range.baseMipLevel = mip;
      b.range.baseArrayLayer = layer;
      images.push_back(b);
    }
  auto cells = image_cells(images);
  coalesce_barriers(images);
  REQUIRE(images.size() == 1);
  REQUIRE(image_cells(images) == cells);
  REQUIRE(images[0].range.levelCount == 4);
  REQUIRE(images[0].range.layerCount == 4);
  auto before = images;
  coalesce_barriers(images);
  REQUIRE(images.size() == 1);
  require_equal(images[0], before[0]);
  images.clear();
  for (unsigned layer : { 0U, 1U }) {
    image_barrier_params b;
    b.range.baseArrayLayer = layer;
    b.src_queue_family = 1;
    b.dst_queue_family = 2;
    images.push_back(b);
  }
  cells = image_cells(images);
  coalesce_barriers(images);
  REQUIRE(images.size() == 2);
  REQUIRE(image_cells(images) == cells);
}
