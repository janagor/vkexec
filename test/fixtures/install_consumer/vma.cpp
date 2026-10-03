#include <vkexec_vma/allocator.hpp>

using move_assign_t = vkexec::vma::allocator &(vkexec::vma::allocator::*)(vkexec::vma::allocator &&) noexcept;
move_assign_t volatile probe = static_cast<move_assign_t>(&vkexec::vma::allocator::operator=);

auto main() -> int { return probe == nullptr; }
