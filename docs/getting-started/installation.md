# Installation

You need CMake 3.29 or newer, a C++20 compiler, a Vulkan loader and a Vulkan-capable device or software driver. Configure from the repository root:

```sh
cmake -S . -B build -G Ninja -DBUILD_TESTING=ON -Dvkexec_BUILD_EXAMPLES=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

CMake fetches pinned source dependencies configured in `Dependencies.cmake`. The full [build guide](../development/building.md) explains options and presets.
