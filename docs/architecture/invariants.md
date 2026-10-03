# Architectural invariants

- Public execution APIs use stdexec vocabulary and semantic sender expressions.
- `vkexec::domain` lowers vkexec operations and delegates unrelated operations to `stdexec::default_domain`.
- Composite lowering makes structural progress and does not add accidental submission boundaries.
- Statically known graphs retain concrete step types; runtime type erasure is isolated to dynamic graphs.
- A successful GPU pass `start()` submits without waiting; blocking is an explicit consumer choice.
- GPU completion transfers to the context scheduler before `after_gpu` and downstream value completion.
- Execution uses Vulkan handles and does not depend on a concrete allocator; VMA is optional.
- Descriptor schema and resource tables do not contain backend-specific heap state.
- Cancellation uses `set_stopped`, failures use `set_error(vkexec::error)`.
- Execution objects retain the shared runtime; resources requiring explicit `context` cleanup are destroyed while the public facade is available.
- Private stdexec APIs, vk-bootstrap types, and extension-specific entry points remain behind their boundaries.
