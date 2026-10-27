#ifndef VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_TRANSACTION_HPP
#define VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_TRANSACTION_HPP

#include <vkexec/detail/resource_state_tracker_validation.hpp>

namespace vkexec::detail {

//! Journals pre-existing mutated resources and barrier-vector append baselines.
//! Newly appended resources are discarded by restoring `buffer_count` / `image_count`.
struct resource_state_tracker_batch_transaction
{
  std::size_t buffer_count{};
  std::size_t image_count{};
  std::size_t dependency_count{};
  std::size_t image_use_count{};
  struct buffer_checkpoint
  {
    std::size_t index{};
    tracked_buffer_resource prior{};
  };
  struct image_checkpoint
  {
    std::size_t index{};
    tracked_image_resource prior{};
  };
  struct step_checkpoint
  {
    std::size_t step{};
    std::size_t buffers_before{};
    std::size_t buffers_after{};
    std::size_t images_before{};
    std::size_t images_after{};
  };
  std::vector<buffer_checkpoint> buffers;
  std::vector<image_checkpoint> images;
  std::vector<step_checkpoint> steps;
};

}// namespace vkexec::detail

#endif// VKEXEC_DETAIL_RESOURCE_STATE_TRACKER_TRANSACTION_HPP
