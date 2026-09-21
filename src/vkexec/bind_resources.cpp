#include <vkexec/bind_resources.hpp>

#include <vkexec/detail/bind_resources.hpp>
#include <vkexec/detail/descriptor_backend.hpp>
#include <vkexec/detail/descriptor_table_backend.hpp>
#include <vkexec/pass.hpp>
#include <vkexec/pipeline.hpp>
#include <vkexec/resource_table.hpp>
#include <vkexec/scheduler.hpp>

#include <cstddef>
#include <span>
#include <utility>

namespace vkexec {

auto bind_resources(pipeline_resources const &pipe, resource_table const &table, std::span<std::byte const> push)
  -> bind_resources_closure
{ return bind_resources_closure{ .pipe = &pipe, .table = table, .push = { push.begin(), push.end() } }; }

auto operator|(schedule_sender snd, bind_resources_closure closure) -> pass_graph_sender
{
  auto step = detail::make_bind_resources_step<detail::set_descriptor_backend>(
    closure.pipe, std::move(closure.table), detail::empty_table_lower_env{}, std::move(closure.push));
  return pass_graph_sender{ .ctx = snd.ctx, .steps = { std::move(step) } };
}

auto operator|(pass_graph_sender graph, bind_resources_closure closure) -> pass_graph_sender
{
  return detail::append_step(std::move(graph),
    detail::make_bind_resources_step<detail::set_descriptor_backend>(
      closure.pipe, std::move(closure.table), detail::empty_table_lower_env{}, std::move(closure.push)));
}

}// namespace vkexec
