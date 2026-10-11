// NOLINTBEGIN(readability-function-cognitive-complexity)

#include <vkexec_extensions/timeline_semaphore/timeline_semaphore.hpp>

volatile decltype(&vkexec::owned::timeline_semaphore::wait) probe = &vkexec::owned::timeline_semaphore::wait;

auto main() -> int { return probe == nullptr; }

// NOLINTEND(readability-function-cognitive-complexity)
