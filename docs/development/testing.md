# Testing

Run `ctest --test-dir build --output-on-failure` after building. CTest includes architecture boundary checks, behavioral tests, and install/subdirectory consumer checks. Vulkan execution tests need a working device or software driver. Enable examples with `-Dvkexec_BUILD_EXAMPLES=ON` to compile the tutorial programs.

Configure with `-Dvkexec_TEST_VALIDATION=ON` to require Vulkan validation layers for contexts created by the shared test helper. The dedicated Linux Vulkan validation CI job installs the layer and builds the unit tests with this setting; ordinary CI jobs use the default `OFF` setting.
