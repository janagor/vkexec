# Design Choices

This template starts a C++ project with safe, modern defaults. Each choice
below explains *why*, so you can keep it, swap it, or turn it off.

## Goals

1. Catch bugs at compile time, not in production.
2. Stay portable across GCC, Clang, MSVC, and Emscripten.
3. Work the same way as a top-level project or as a subdirectory dependency.

## Layout

| File | Role |
| --- | --- |
| `CMakeLists.txt` | Top-level wiring. |
| `ProjectOptions.cmake` | All `myproject_*` options and setup macros. |
| `Dependencies.cmake` | CPM package fetch, gated by `if(NOT TARGET ...)`. |
| `cmake/*.cmake` | One concern per file (warnings, sanitizers, hardening, ...). |

`PROJECT_IS_TOP_LEVEL` flips defaults: strict when you own the build, quiet
when you are a dependency.

## C++ standard

C++23, set only if a parent project has not chosen one. `CMAKE_CXX_EXTENSIONS`
is off so the standard flag is `-std=c++23`, not `-std=gnu++23`. This avoids
`-Wpedantic` conflicts with precompiled headers.

## Warnings

`cmake/CompilerWarnings.cmake` enables a curated set per compiler — `/W4`
plus extras on MSVC, and `-Wall -Wextra -Wshadow -Wconversion -Wpedantic ...`
on GCC/Clang. Top-level builds add `-Werror` / `/WX`. Source:
[cppbestpractices](https://github.com/lefticus/cppbestpractices/blob/master/02-Use_the_Tools_Available.md).

## Sanitizers

ASan and UBSan are on by default for top-level GCC/Clang builds when a link
probe shows them working. TSan, LSan, and MSan are off — they conflict with
each other and MSan needs an instrumented standard library. Emscripten and
MSVC skip the sanitizer pass.

## Hardening

`cmake/Hardening.cmake` adds `_FORTIFY_SOURCE=3` (release builds),
`_GLIBCXX_ASSERTIONS`, `-fstack-protector-strong`, `-fcf-protection`, and
`-fstack-clash-protection` when supported. MSVC gets `/sdl /DYNAMICBASE
/guard:cf /NXCOMPAT /CETCOMPAT`. When no full sanitizer is active, the UBSan
minimal runtime is layered on top.

## Static analysis

clang-tidy and cppcheck run as part of the build, on by default at top level.
They are separate options because one tool may not be installed in every
environment.

## Link-time optimization

IPO/LTO is on by default at top level. It is gated through
`check_ipo_supported` so unsupported toolchains skip it.

## Dependencies

[CPM](https://github.com/cpm-cmake/CPM.cmake) fetches sources at configure
time. Each package is gated by `if(NOT TARGET ...)`, so a parent project can
supply its own version. `SYSTEM YES` silences warnings from third-party
headers. Core dependencies are Catch2, Boost.System, stdexec, vk-bootstrap, and VMA.
glslang is fetched only when `vkexec_BUILD_TOOLS=ON`; GLFW is fetched only for examples.
`vkexec::vkexec` propagates both `Vulkan::Headers` and `Vulkan::Vulkan` to consumers.

## Libraries

* `vkexec` — compute runtime (`context`, `buffer`, `compute_pipeline`, `compute_pass`)
* `vkexec_graphics` — backend-neutral presenter, graphics pipelines, and `draw` senders
* `vkexec_tools` (`vkexec::tools`) — optional GLSL-to-SPIR-V compilation and GLSL pipeline factories

## Descriptor strategies

Descriptor handling has three independent layers:

1. `descriptor_schema<...>` is the compile-time shader contract. Entries are
   `storage_buffer<Slot>`, `storage_image<Slot>`, `sampled_image<Slot>`, and
   `sampler_binding<Slot>` (`sampler` remains the owning Vulkan wrapper). A schema
   validates sorted, unique logical slots, builds a `resource_table` with checked
   arity, and derives classic `layout_desc` values through `layout_desc_from_schema`.
2. `resource_table` is the core, heap-agnostic runtime bag of logical buffer,
   image-view, and sampler bindings. It contains Vulkan handles, sizes, and layouts,
   but no descriptor-heap metadata.
3. Private descriptor backends own pipeline layout/flags, command recording, and
   lowering into backend-specific bound values. Public callers select only the
   `descriptor_sets` or `descriptor_heap` strategy.

Schema-derived classic layouts retain explicit descriptor binding slots. Existing
hand-written `layout_desc` callers leave `binding_slots` empty and continue to use
positional bindings.

Descriptor sets lower with an empty core environment. Descriptor-heap resource and
sampler indices, mapped bytes, descriptor sizes/strides, and image/sampler create-info
metadata remain extension-only in `heap_table_lower_env`. `lower_and_bind_push`
composes the two backend concepts without adding heap knowledge to schema or table
types. `bind_resources` exposes that operation as a pass-graph step; its default
backend is descriptor sets, while the extension overload is selected with
`descriptor_heap`.

## Public and private headers

Installed headers live under `include/` and contain the supported API. Template
machinery may use a local `detail` namespace inside its owning public header, but
detail types do not appear in public signatures. Shared implementation headers
live under `src/**/detail/`, are supplied through the non-exported
`vkexec_private_headers` target, and are never installed.

Public algorithm verbs are backend-neutral. Compute and dynamic-rendering graphics
select heap behavior once with `descriptor_heap`, then use `create_compute_resources`,
`factory::compute_pipeline`, `create_graphics_resources`, `factory::graphics_pipeline`,
`record_pass`, `record_draw`, and `compute_pass`. Literal heap mechanism APIs such as
`descriptor_heap_buffer`, `cmd_bind_resource_heap`, `cmd_bind_sampler_heap`, and
descriptor writers keep their names.

The core intentionally provides neither set-to-heap emulation nor a descriptor-heap
slot allocator. Applications own physical heap indices and pass them explicitly in
`heap_table_lower_env`.

## Testing

* `test/tests.cpp` — Catch2 unit tests.
* `test/constexpr_tests.cpp` — the same checks at compile time, so bugs
  become build errors.

## Targets and packaging

`myproject_options` and `myproject_warnings` are `INTERFACE` libraries that
hold flags. Real targets link them to inherit the configuration without
touching global state. `CPack` package names embed compiler, version, and
short Git SHA, so a binary maps to one build.

## Defaults for daily use

The default build type is `RelWithDebInfo` — debuggable and fast.
`compile_commands.json` is always exported, for editors and clang tooling.

## Changing the defaults

Every knob is a CMake option named `myproject_ENABLE_<feature>`. Flip it on
the configure line, for example:

    cmake -B build -S . -Dmyproject_ENABLE_CLANG_TIDY=OFF

The `myproject_` prefix is the placeholder the rename workflow replaces, so
renaming the project is one search-and-replace.
