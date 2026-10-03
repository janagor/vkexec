#include <vkexec_extensions/dynamic_rendering/rendering.hpp>

volatile decltype(&vkexec::cmd_end_rendering) probe = &vkexec::cmd_end_rendering;

auto main() -> int { return probe == nullptr; }
