# Pass graphs

A pass graph is a collection of GPU pass steps with explicit dependencies. Static composition keeps the concrete step types in `pass_graph_sender<Steps...>`; this lets several adjacent operations lower into one graph without an accidental submission boundary. A dynamic pass graph is available when the graph structure is genuinely chosen at runtime.

`connect()` transfers the graph into an operation state with a receiver. During `start()`, the graph plans its batches and records steps into graph-owned command buffers according to their dependencies. A barrier adaptor between passes describes a GPU memory and execution dependency, so a compute → barrier → compute sequence can remain one submission.

The compiled [multi-pass program](../examples/multi-pass.md) shows this sequence. For how semantic operations become primitive steps, see [Domain lowering](../architecture/domain-lowering.md).

## Typed branches

`vkexec::when_all()` composes a GPU graph and is distinct from `stdexec::when_all`, which combines sender completions and values. It accepts two or more pass adaptor closures, and each branch must contain at least one pass. Each branch starts after the same incoming frontier. Its final pass becomes a frontier node; the next ordinary pass depends on every branch's frontier. The adaptor adds no GPU join submission. Nested branches work the same way, and `on_queue()` inside a branch applies only to that branch. A pass after `when_all()` keeps the queue selected before the branches.

```cpp
auto graph = stdexec::schedule(ctx.get_scheduler())
             | root_pass()
             | vkexec::when_all(
                 vkexec::on_queue(compute) | compute_pass_a() | compute_pass_b(),
                 vkexec::on_queue(graphics) | graphics_pass())
             | composite_pass();
```

Here `composite_pass()` depends on both `compute_pass_b()` and `graphics_pass()`. If the graph ends at `when_all()`, the existing terminal join waits for both branches. Presentation remains an outer graph boundary; place `present()` after the branches and any final graphics pass.

## Runtime branches

Use `dynamic_pass_graph_sender::append_after()` when passes form a DAG. Nodes are added in topological order. The method returns a node index; pass the indices of required predecessors to later nodes. An empty predecessor list starts an independent branch. A node with multiple predecessors joins branches. `append()` and pipe composition retain their usual sequential dependency when used after DAG nodes.

Set queue affinity with `append_queue()` before adding each node. Resource declarations on passes let the planner add missing dependencies for conflicting uses, including queue ownership transfers. The executor submits batches without a host wait. When several branches remain terminal, an internal join submission waits for all of them before the sender completes. With presentation, the final graphics batch serves as that join.

Compatible read-only uses can run on different queues in the same family when they preserve the resource layout. A later write waits for every reader in the frontier. Layout transitions and queue-family ownership changes establish a new ordered access; ownership transfers use one reader as the release handoff.

## Buffer ranges

Buffer declarations track half-open byte intervals independently. For example, writing `[0, 256)` and then reading `[128, 384)` adds a dependency and a barrier for `[128, 256)` only. The remaining bytes of the read are first uses. Disjoint ranges of one buffer add no resource dependency, including disjoint declarations within a single pass. Compatible overlapping reads in one pass are also allowed when they do not require a layout or ownership transition; overlapping accesses involving a write must be combined in one declaration, such as `read_write` for storage access.

`VK_WHOLE_SIZE` means from the declared offset to the end of the buffer. Without a registered extent, the planner keeps that end symbolic and rejects later finite cuts of that open interval, as well as open suffixes created when a whole-size use extends past already-tracked finite coverage. Register the extent when mixing whole-size declarations with finite ranges that can reach the actual resource end, as explained below. Zero-length ranges and overflowing finite ends are rejected. The application remains responsible for keeping declarations within the actual buffer and accurately describing recorded commands.

Each interval retains its producer, outstanding readers, and owning queue family when split. Adjacent intervals merge only when their complete synchronization states match. Queue-family release and acquire barriers cover the same transferred interval. Unregistered resources default to exclusive sharing. Extent declarations can report concurrent sharing; those resources retain data dependencies and barriers without queue-family ownership transfers. Each newly accessed interval must already be owned by its first graph queue family, with any imported work synchronized externally.

Dependencies remain between graph steps. Disjoint buffer regions avoid additional resource edges, but explicit graph ordering still applies and concurrent GPU execution is not guaranteed. Tracking is by Vulkan handle; memory aliasing between distinct handles is not detected.

## Image subresources

Image declarations track rectangles of mip levels and array layers within an aspect. A pass may read mip 0 while writing mip 1 of the same image. Partial overlaps split existing regions and retain the old layouts, owners, producers, and readers of every piece. A declaration spanning different old layouts generates separate transitions for those regions. `VK_REMAINING_MIP_LEVELS` and `VK_REMAINING_ARRAY_LAYERS` keep symbolic ends, like `VK_WHOLE_SIZE` for buffers.

Every previously untracked subresource needs a known `initial_layout` before a read or `read_write`. A read spanning tracked and untracked regions must supply it for the uncovered parts. Tracked parts always use their tracked layout. A first write can discard contents from `VK_IMAGE_LAYOUT_UNDEFINED`. `layout` overrides and render-pass `final_layout` values affect only the declared region. Dynamic-rendering recorders declare their attachment regions through `custom_pass`; render-pass graphics steps use `graphics_attachment::range` and the supplied render-pass layouts.

A conflicting overlap in one pass is rejected before its declaration changes tracked state or the synchronization plan. Disjoint declarations are allowed. Compatible reads may overlap within a pass if neither requires a new transition. Resource declarations do not describe ordering inside a recorder; combine a storage read and write in one `read_write` use and supply any required internal synchronization in the recorder.

### Supported aspect and image cases

- Color images support independent mip and array-layer ranges.
- A combined depth/stencil mask stays coupled in every generated barrier. Use this form consistently for depth/stencil images without `separateDepthStencilLayouts`.
- Separate depth and stencil masks are independent. For a format containing both aspects, the caller must enable `separateDepthStencilLayouts` and choose valid layouts. Depth-only and stencil-only formats use their respective aspect. Mixing combined and separate depth/stencil declarations for one handle is rejected because declarations do not carry the format or enabled-feature metadata needed to reconcile them.
- Multiplanar images may use `VK_IMAGE_ASPECT_COLOR_BIT` to describe all planes together. Plane and memory-plane aspect declarations are rejected; the tracker does not model their aliasing with color.
- A 3D image declaration describes barrier subresources, not the layer coordinates of a 2D slice view. Without `maintenance9`, declare all depth slices of each affected mip using `baseArrayLayer = 0` and `layerCount = 1` (or the equivalent remaining range). Independently tracked slice-view ranges and their implicit render-pass transitions are unsupported. The caller must supply the full affected mip range; the tracker has no image-type metadata to expand view coordinates.

These restrictions follow Vulkan's [image memory barrier](https://docs.vulkan.org/refpages/latest/refpages/source/VkImageMemoryBarrier2.html) and [subresource range](https://docs.vulkan.org/refpages/latest/refpages/source/VkImageSubresourceRange.html) rules. Object bounds, format compatibility, and feature-dependent validity remain the application's responsibility and are checked by Vulkan validation. As with buffers, unregistered images default to exclusive sharing; registered concurrent images omit ownership transfers. Imports must already be synchronized. Aliasing between separate image or buffer handles is not tracked.

## Optional extent registration

Include `buffer_extent` or `image_extent` in the first pass's `uses(...)` to supply actual dimensions. Extents are registered before that pass's accesses, regardless of their argument order. The existing `read`, `write`, and `read_write` overloads are unchanged.

```cpp
auto resources = vkexec::uses(
  vkexec::buffer_extent{ .buffer = buffer, .size = buffer_bytes },
  vkexec::image_extent{ .image = image, .mip_levels = mip_levels, .array_layers = array_layers },
  vkexec::write(buffer, vkexec::buffer_usage::transfer_destination));
```

Registration must precede the handle's first use in planning order. Repeating identical metadata is allowed; conflicting or late registration fails. For passes with inferred resources, an earlier `custom_pass` can carry extent-only declarations and an empty recorder. The internal tracker also exposes `register_extent(...)` for direct planning clients.

Registered bounds resolve whole-size and remaining-count ranges to finite ends and reject out-of-bounds declarations. A finite cut of an unregistered open range is rejected, because the planner cannot prove the leftover symbolic suffix is nonempty and a barrier beginning at the true resource end is invalid Vulkan. The same rule applies when a later whole-size or remaining-count use would invent an uncovered open gap past finite tracked coverage. For images, that open-gap check is per mip or layer dimension against tracked coverage that shares aspects and the other dimension, so disjoint finite mip ranges may still share `VK_REMAINING_ARRAY_LAYERS` without registration. Without registration, use consistent symbolic ranges or only finite ranges. `uses(...)` applies every declaration in the batch atomically and rolls back only the resources and synchronization appends touched by that batch if any declaration fails.

Both extent types have `sharing_mode`, defaulting to `VK_SHARING_MODE_EXCLUSIVE`. Set it to `VK_SHARING_MODE_CONCURRENT` only when the resource was created that way. The caller must ensure every graph queue family belongs to the creation-time sharing list. Registration supplies dimensions and sharing mode, not image format, view type, memory aliasing, or feature availability.
