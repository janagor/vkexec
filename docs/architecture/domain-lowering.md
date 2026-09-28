# Domain lowering

```{graphviz} diagrams/lowering.dot
```

## `vkexec::domain`

`domain::transform_sender()` calls `lower_vkexec_sender` when vkexec has a lowering for the operation. Otherwise it delegates to `stdexec::default_domain`, so ordinary stdexec algorithms remain composable.

## Semantic sender expressions

An algorithm such as `compute_pass(...)` returns a stdexec sender adaptor closure. Piping it onto a child sender creates an expression shaped as `sender_expr<Tag, Data, Child>`. Environment queries forward to the child until the domain transforms it. Construction does not record or submit Vulkan work.

Public composable algorithms should produce semantic expressions when the vkexec domain can lower them. Use `stdexec::sender_adaptor_closure` rather than a separate `operator|` mechanism.

## Domain lowering

```text
algorithm(args...) -> expr_closure<Tag, Data>
  -> sender_expr<Tag, Data, Child>
  -> domain::transform_sender()
  -> normalize_vkexec_expression()
  -> lower_vkexec_pass_step()
  -> typed Step -> pass_graph_sender<Steps...>
```

`lower_vkexec_expression` expands a composite operation into other semantic expressions. Each expansion must make structural progress. Normalization completes before primitive steps are collected, so a composite does not accidentally create a submission boundary. `lower_vkexec_pass_step` turns a primitive operation into one recordable step. A `schedule_sender` predecessor materializes directly into a pass graph without an extra `let_value`; other scheduler-completing predecessors use `let_value` to preserve their completion dependency.
