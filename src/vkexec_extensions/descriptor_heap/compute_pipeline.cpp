#include <vkexec_extensions/descriptor_heap/compute_pipeline.hpp>

#include <vkexec/pass.hpp>
#include <vkexec_extensions/descriptor_heap/pass.hpp>
#include <vkexec_extensions/descriptor_heap/strategy.hpp>

namespace vkexec {

auto compute_pass([[maybe_unused]] descriptor_heap_t strategy, compute_bind bind, dispatch groups)
  -> descriptor_compute_pass_closure
{ return descriptor_compute_pass_closure{ .inner = compute_pass(bind, groups) }; }

auto compute_pass([[maybe_unused]] descriptor_heap_t strategy, compute_bind bind, indirect_dispatch groups)
  -> descriptor_compute_pass_closure
{ return descriptor_compute_pass_closure{ .inner = compute_pass(bind, groups) }; }

}// namespace vkexec
