#ifndef VKEXEC_EXECUTION_HPP
#define VKEXEC_EXECUTION_HPP

//! \file
//! Execution umbrella: stdexec scheduler, pass graphs, and borrowable Vulkan handles.
//!
//! Prefer this for embedders that adopt a device and record with `compute_bind`.
//! Owning RAII types live in `<vkexec/resources.hpp>`.

#include <vkexec/barrier.hpp>
#include <vkexec/bind_resources.hpp>
#include <vkexec/context.hpp>
#include <vkexec/copy.hpp>
#include <vkexec/descriptor_schema.hpp>
#include <vkexec/descriptor_strategy.hpp>
#include <vkexec/device_procs.hpp>
#include <vkexec/domain.hpp>
#include <vkexec/error.hpp>
#include <vkexec/error_helpers.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/push.hpp>
#include <vkexec/queue_submit.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/result.hpp>
#include <vkexec/scheduler.hpp>
#include <vkexec/sender.hpp>
#include <vkexec/submit.hpp>
#include <vkexec/submit_scope.hpp>
#include <vkexec/sync_wait.hpp>
#include <vkexec/sync_wait_outcome.hpp>
#include <vkexec/tensor_sync.hpp>
#include <vkexec/vulkan_requirements.hpp>

namespace vkexec {}// namespace vkexec

#endif// VKEXEC_EXECUTION_HPP
