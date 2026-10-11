// NOLINTBEGIN(readability-function-cognitive-complexity)

#include <vkexec/error.hpp>

#include <string_view>

auto main() -> int { return std::string_view{ vkexec::category().name() } == "vkexec" ? 0 : 1; }

// NOLINTEND(readability-function-cognitive-complexity)
