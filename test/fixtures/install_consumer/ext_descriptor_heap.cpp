// NOLINTBEGIN(readability-function-cognitive-complexity)

#include <vkexec_extensions/descriptor_heap/descriptor_heap.hpp>

volatile decltype(&vkexec::descriptor_heap_byte_size) probe = &vkexec::descriptor_heap_byte_size;

auto main() -> int { return probe == nullptr; }

// NOLINTEND(readability-function-cognitive-complexity)
