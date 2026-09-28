# Public and private boundaries

## Public headers

`include/vkexec/*.hpp` provides the supported core API. Optional modules have their own include trees and namespaces.

## `include/vkexec/detail`

Installed detail headers support public template implementation but are not public API. Detail types must not become public signature commitments.

## `src/**/detail`

Compiled implementation headers remain private and are not installed.

## stdexec private API quarantine

vkexec uses the public `<stdexec/execution.hpp>` vocabulary. Direct `stdexec::__*`, `ex::__*`, `exec::__*`, `STDEXEC::__*`, and `stdexec/__detail/*` dependencies are forbidden outside `include/vkexec/detail/stdexec_compat.hpp`. That compatibility header must not expose private stdexec types in public signatures.

## vk-bootstrap boundary

vk-bootstrap is an implementation mechanism for instance/device discovery and creation. Public APIs use Vulkan handles and vkexec types, not vk-bootstrap types. Applications with their own device setup use `factory::adopt_context()`.

## Optional dependency boundaries

VMA stays in `vkexec_vma`; extension implementation state and entry points stay in their extension modules. Core allocation and descriptor semantics do not depend on either mechanism.
