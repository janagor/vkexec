# Synchronization

## Pass-local barriers

Barrier adaptors describe GPU memory and execution dependencies inside a pass graph. A compute/barrier/compute chain records the barrier between the two dispatches in one ordered submission.

## synchronization2 versus legacy lowering

A semantic barrier uses synchronization2 Flags2 and `vkCmdPipelineBarrier2` when available. Otherwise it uses checked translation to the legacy barrier command. Unsupported legacy translations fail through the pass graph's error channel. Adopted contexts use feature information supplied by the embedder.

## Queue submission

Recording and queue submission are separate from barrier semantics. The context serializes command-pool and queue access as required. A submitted graph uses a fence-backed completion sender.

## GPU completion versus host scheduling

The fence establishes GPU completion. `continues_on(ctx->get_scheduler())` then establishes the host context for cleanup and downstream value completion. A pass-local barrier does not serve this host/GPU completion role.
