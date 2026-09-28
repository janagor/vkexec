# Testing

Run `ctest --test-dir build --output-on-failure` after building. CTest includes architecture boundary checks, behavioral tests, and install/subdirectory consumer checks. Vulkan execution tests need a working device or software driver. Enable examples with `-Dvkexec_BUILD_EXAMPLES=ON` to compile the tutorial programs.
