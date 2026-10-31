#include <csignal>
#include <cstdlib>
#include <exception>
#include <utility>

#include <stdexec/execution.hpp>
#include <vkexec/context.hpp>
#include <vkexec/record_into.hpp>
#include <vkexec/result.hpp>
#include <vkexec/submit_scope.hpp>
#include <vkexec/sync_wait.hpp>

namespace {
constexpr int k_context_unavailable_exit_code = 125;
constexpr int k_retirement_guard_exit_code = 77;
}// namespace

// Fresh process: Vulkan drivers may start threads, so never fork a live context.
auto main() -> int
{
  auto made = vkexec::try_sync_wait_value(vkexec::factory::make_context());
  if (!made) { return k_context_unavailable_exit_code; }
  auto ctx = std::move(*made);
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  if (!opened) { return EXIT_FAILURE; }
  auto scope = vkexec::expected_take(opened);
  auto recording = vkexec::record_into(*ctx, scope.cmd, stdexec::schedule(ctx->get_scheduler()), { scope.queue });
  if (!recording) { return EXIT_FAILURE; }
  std::set_terminate([]() -> void { std::_Exit(k_retirement_guard_exit_code); });
  (void)std::signal(SIGABRT, [](int signal_number) -> void {
    (void)signal_number;
    std::_Exit(k_retirement_guard_exit_code);
  });
  // Intentionally leave recording active. Both the debug assertion and the
  // release termination guard must stop the process with the sentinel code.
  return EXIT_SUCCESS;
}
