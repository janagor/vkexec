#include <vkexec_extensions/descriptor_heap/compute_pipeline.hpp>

#include <vkexec/pass.hpp>

#include <cstdint>
#include <vkexec_extensions/descriptor_heap/heap_compute_pipeline.hpp>
#include <vkexec_extensions/descriptor_heap/pass.hpp>

namespace vkexec {

auto compute_heap_pass(compute_bind bind, dispatch groups) -> heap_compute_pass_closure
{ return heap_compute_pass_closure{ .inner = compute_pass(bind, groups) }; }

auto compute_heap_pass(compute_bind bind, indirect_dispatch groups) -> heap_compute_pass_closure
{ return heap_compute_pass_closure{ .inner = compute_pass(bind, groups) }; }

auto compute_heap_pass(heap_compute_pipeline const &pipe, std::uint32_t work_count) -> heap_compute_pass_closure
{ return compute_heap_pass(pipe.bind(), pipe.groups_for(work_count)); }

auto compute_heap_pass(heap_compute_pipeline const &pipe, indirect_dispatch groups) -> heap_compute_pass_closure
{ return compute_heap_pass(pipe.bind(), groups); }

}// namespace vkexec
