#ifndef VKEXEC_VKEXEC_HPP
#define VKEXEC_VKEXEC_HPP


#include <vkexec/barrier.hpp>
#include <vkexec/buffer.hpp>
#include <vkexec/bulk.hpp>
#include <vkexec/compute_pipeline.hpp>
#include <vkexec/context.hpp>
#include <vkexec/descriptor_heap.hpp>
#include <vkexec/device_procs.hpp>
#include <vkexec/domain.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec/frame_ring.hpp>
#include <vkexec/gpu_buffer.hpp>
#include <vkexec/image.hpp>
#include <vkexec/image_view.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/push.hpp>
#include <vkexec/push_data.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/rendering.hpp>
#include <vkexec/sampler.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/submit.hpp>
#include <vkexec/timeline_semaphore.hpp>
#include <vkexec/vulkan_requirements.hpp>
#include <vkexec_edsl/edsl.hpp>

namespace vkexec {
// Public umbrella header for the vkexec stdexec Vulkan backend.
}

#endif// VKEXEC_VKEXEC_HPP
