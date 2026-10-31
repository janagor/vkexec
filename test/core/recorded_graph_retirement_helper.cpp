#include <csignal>
#include <cstdlib>
#include <exception>
#include <vkexec/record_into.hpp>
#include <vkexec/sync_wait.hpp>

// Fresh process: Vulkan drivers may start threads, so never fork a live context.
auto main() -> int
{
  auto made = vkexec::try_sync_wait_value(vkexec::factory::make_context());
  if (!made) { return 125; }
  auto ctx = std::move(*made);
  auto opened = vkexec::detail::submit_scope::open(*ctx);
  if (!opened) { return EXIT_FAILURE; }
  auto scope = vkexec::expected_take(opened);
  auto recording = vkexec::record_into(*ctx, scope.cmd, stdexec::schedule(ctx->get_scheduler()), { scope.queue });
  if (!recording) { return EXIT_FAILURE; }
  std::set_terminate([] { std::_Exit(77); });
  std::signal(SIGABRT, [](int) { std::_Exit(77); });
  // Intentionally leave recording active. Both the debug assertion and the
  // release termination guard must stop the process with the sentinel code.
  return EXIT_SUCCESS;
}
