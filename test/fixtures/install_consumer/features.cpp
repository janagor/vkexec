#include <vkexec_features/timeline_semaphore.hpp>

volatile decltype(&vkexec::feat::feature_traits<vkexec::feat::timeline_semaphore>::available) probe =
  &vkexec::feat::feature_traits<vkexec::feat::timeline_semaphore>::available;

auto main() -> int { return probe == nullptr; }
