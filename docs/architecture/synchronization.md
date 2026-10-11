# Synchronization

## Pass-local barriers

Barrier adaptors describe GPU memory and execution dependencies inside a pass graph. A compute/barrier/compute chain records the barrier between the two dispatches in one ordered submission.

## synchronization2 versus legacy lowering

A semantic barrier uses synchronization2 Flags2 and `vkCmdPipelineBarrier2` when available. Otherwise it uses checked translation to the legacy barrier command. Unsupported legacy translations fail through the pass graph's error channel. Adopted contexts use feature information supplied by the embedder.

## Queue submission

Recording and queue submission are separate from barrier semantics. The context serializes command-pool and queue access as required. A submitted graph uses a fence-backed completion sender.

## GPU completion versus host scheduling

The fence establishes GPU completion. `continues_on(ctx->get_scheduler())` then establishes the host context for cleanup and downstream value completion. A pass-local barrier does not serve this host/GPU completion role.

## Resource regions

The resource-state tracker owns a collection of non-overlapping regions for each Vulkan handle. Buffer regions are sorted half-open intervals; image regions are aspect-specific mip/layer rectangles, with a combined depth/stencil aspect group retained when declarations require coupled layouts. Missing regions represent first graph uses. Open ends represent Vulkan's whole-size and remaining-count sentinels; optional extent registration resolves them to known finite ends before partitioning. Finite cuts of unregistered open ranges, and open uncovered gaps that would extend finite tracked coverage into a symbolic suffix, are rejected rather than retaining an unproven empty tail. Image open-gap rejection is scoped to overlapping aspects and the other subresource dimension, so independent mip or aspect strips may share a symbolic remaining-count bound. Atomic `uses(...)` batches journal only mutated resources and append baselines instead of cloning the full step plan.

Each region has an access frontier: an optional synchronization anchor and every outstanding compatible reader. Splitting copies the whole frontier and queue-family owner, plus the image layout. Only intersections advance; uncovered pieces use first-use behavior. Validation runs across all affected pieces before dependencies, barriers, or persistent state change. Graph construction supplies declarations in step order.

Dependencies remain step edges and are deduplicated in the existing plan vector. A conflicting access joins the relevant frontier. Exclusive-resource family transfers choose a reader handoff, order the other readers before it, and emit matching release/acquire ranges. The release scope includes all frontier accesses on the handoff queue; accesses on other queues are ordered by submission dependencies. Registered concurrently shared resources synchronize data and layouts without ownership transfers.

Merging compares owner, image layout, and complete anchor/reader metadata, including step, access kind, queue, stage masks, and access masks. Image unions must be rectangular. Plan finalization can coalesce adjacent ordinary barriers with identical synchronization parameters and old/new layouts. Ownership-transfer barriers retain their paired ranges. No state is discarded to cap region counts.

`resource_sync_plan::image_access` retains the declared subresource range along with the usage, final layout, and step. Presentation validation checks that declarations describe the swapchain color subresource; barrier recording already consumes each barrier's explicit range.

The optional [planning benchmark](../development/testing.md#resource-planning-benchmark) measures fragmentation and plan size before introducing more complex lookup or dependency structures.
