#include <vkexec_graphics/presenter.hpp>

volatile decltype(&vkexec::owned::presenter::wait_idle) probe = &vkexec::owned::presenter::wait_idle;

auto main() -> int { return probe == nullptr; }
