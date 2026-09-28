# Creating a context

Use `factory::make_context` when vkexec should create the Vulkan instance and device. Configure validation and device requirements before starting GPU work. Retain the context until all senders and context-dependent resources finish.
