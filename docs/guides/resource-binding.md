# Resource binding

Use a descriptor schema or layout description to express shader bindings. A resource table holds logical buffer, image-view, and sampler handles. Bind resources before composing the pass sender; keep the referenced Vulkan objects and descriptor state alive through completion. Choose a descriptor strategy according to the backend.
