#include <vkexec/error.hpp>

volatile decltype(&vkexec::category) probe = &vkexec::category;

auto main() -> int { return probe == nullptr; }
