# VMA allocation

Enable `vkexec_BUILD_VMA` and link the `vkexec_vma` target. `vma::factory::make_allocator(*ctx)` creates an owning allocator; `vma::factory::adopt_allocator(*ctx, external_vma)` borrows an existing one. Allocation returns senders. Keep resources and allocator alive through completion, then destroy them before the context.
