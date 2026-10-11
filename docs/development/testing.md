# Testing

Run `ctest --test-dir build --output-on-failure` after building. CTest includes architecture boundary checks, behavioral tests, and install/subdirectory consumer checks. Vulkan execution tests need a working device or software driver. Enable examples with `-Dvkexec_BUILD_EXAMPLES=ON` to compile the tutorial programs.

Configure with `-Dvkexec_TEST_VALIDATION=ON` to require Vulkan validation layers for contexts created by the shared test helper. The dedicated Linux Vulkan validation CI job installs the layer and builds the unit tests with this setting; ordinary CI jobs use the default `OFF` setting.

## Resource-region tests

The core tests tagged `[ranges]` exercise buffer interval and image rectangle algebra, preserved frontiers, independent regions, exact barrier coverage, invalid-declaration rollback, rejection of unregistered open-range splits and open uncovered suffixes, registered extent clamping, atomic `uses(...)` batches (including registration and ownership-release rollback), exclusive/concurrent sharing, ownership transfers, layout overrides, and conservative coalescing without a Vulkan device. Deterministic randomized cases compare buffer plans against per-byte state and image plans against a small aspect/mip/layer grid. They check dependency edges, source scopes, old/new layouts, and uncovered or unintended barrier coverage. The optional resource-region benchmark includes a many-pass `uses(...)` batch case for planning overhead.

VMA integration tests tagged `[gpu][ranges]` verify actual readback data for partially overlapping buffer suballocations and a mipmap chain. The dynamic-rendering integration test renders one mip while preserving another and checks both outputs. The graphics tests also verify that implicit render-pass final layouts remain scoped to their attachment ranges.

After building, select these tests with `ctest --test-dir build -R 'buffer interval|buffer suballocations|mipmap graph|image rectangles|dynamic rendering to one mip' --output-on-failure`, or run the complete test suite. Configure `vkexec_TEST_VALIDATION=ON` and use Lavapipe where a hardware Vulkan device is unavailable. Set `VK_LAYER_VALIDATE_SYNC=1` for synchronization validation; older validation layers use `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`. The Linux validation job sets both and fails on validation or synchronization-hazard messages. See the validation layer's [settings migration guide](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/main/docs/updating_from_VK_EXT_validation_features.md).

## Resource planning benchmark

Configure with `-Dvkexec_BUILD_BENCHMARKS=ON`, build `vkexec_resource_regions_benchmark`, and run `build/benchmark/vkexec_resource_regions_benchmark`. This optional [Google Benchmark](https://github.com/google/benchmark) executable uses opaque test handles and does not create a Vulkan device. Use `--benchmark_format=csv` for CSV output.

Reported counters include peak and final region counts, barrier counts, and dependency counts. Scenarios cover 512 adjacent buffer writes, alternating overlaps, repeated identical buffer ranges, a mipmap chain, 256 individually accessed image layers, and a many-pass `uses(...)` batch run at 64, 128, 256, 512, and 1024 steps to observe planning scaling as one buffer accumulates regions. Timing includes partitioning, frontier updates, state merging, and final barrier coalescing. Region growth from genuinely different synchronization histories is retained deliberately. Compare measurements before changing the sorted-vector representation or linear dependency deduplication.
