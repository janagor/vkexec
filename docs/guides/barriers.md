# Barriers

Place `barrier::compute_to_compute()` between passes with dependent accesses. It records a GPU memory and execution barrier in the same pass graph. Synchronization2 is used when available, with checked legacy lowering otherwise. See the [compiled pass example](../examples/multi-pass.md).
