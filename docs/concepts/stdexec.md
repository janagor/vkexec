# stdexec composition

vkexec uses stdexec senders, sender adaptors, receivers, and completion channels. `stdexec::schedule(ctx->get_scheduler())` provides a sender whose environment advertises both the vkexec domain and the scheduler for value completion. Pass adaptors such as `compute_pass` and barriers pipe onto that predecessor.

The domain interprets vkexec semantic operations. Unrelated stdexec algorithms are delegated to the default domain, so ordinary sender composition remains available. A pass adaptor requires a predecessor whose value completion is guaranteed on the vkexec scheduler; a domain-only environment does not claim that scheduling guarantee.

Successful pass work reaches `set_value` after GPU completion and post-GPU actions. Expected failures reach `set_error(vkexec::error)`; cancellation reaches `set_stopped`. Error and stopped completion are not promised to run on the same scheduler as value completion. See [Completion](completion.md) and [Domain lowering](../architecture/domain-lowering.md).
