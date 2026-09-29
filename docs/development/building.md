# Building

The root `CMakeLists.txt` requires CMake 3.29 and each installed library target advertises C++20. You also need a C++20 compiler, a Vulkan loader, and a device or software driver for GPU tests. CMake fetches pinned source dependencies specified in `Dependencies.cmake`; the first configure needs network access unless those dependencies are already available.

## Configure a local build

From the repository root:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON -Dvkexec_BUILD_EXAMPLES=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Choose a different build directory for Release or sanitizer configurations. CMake selects `CC` and `CXX` on the first configure; use a fresh directory when changing compilers.

## Use a preset

`CMakePresets.json` defines GCC and Clang Debug/Release presets on Unix-like systems and MSVC/Clang presets on Windows. For example:

```sh
cmake --preset unixlike-clang-debug
cmake --build out/build/unixlike-clang-debug
ctest --preset test-unixlike-clang-debug
```

A preset can enable more tooling than the minimal command above. Inspect its cache variables in `CMakePresets.json` before using it in an environment without those tools.

## Select modules

| Option | Purpose |
|---|---|
| `vkexec_BUILD_EXAMPLES` | Compile the programs in `examples/` |
| `vkexec_BUILD_TOOLS` | Build optional GLSL/SPIR-V tooling; requires glslang |
| `vkexec_BUILD_VMA` | Build VMA-backed allocation helpers |
| `VKEXEC_ENABLE_EXCEPTIONS` | Select exception or outcome based `sync_wait` behavior |
| `vkexec_ENABLE_CLANG_TIDY`, `vkexec_ENABLE_CPPCHECK` | Run static analysis during the build |
| `vkexec_ENABLE_COVERAGE` | Enable coverage instrumentation |

Defaults vary between top-level development and use as a subproject. `ProjectOptions.cmake` is authoritative for the complete option list. See [Dependencies](dependencies.md), [Testing](testing.md), and [Development environments](environments.md).

## Install and consume the package

After building, install vkexec into a chosen prefix:

```sh
cmake --install build --prefix /path/to/vkexec-prefix
```

For a multi-configuration generator, add `--config Debug` or `--config Release` to match the configuration you built. The installed package exports its public targets and headers. In a separate consumer project, locate the config package and link the core target:

```cmake
cmake_minimum_required(VERSION 3.29)
project(vkexec_consumer LANGUAGES CXX)

find_package(vkexec CONFIG REQUIRED)
add_executable(app main.cpp)
target_link_libraries(app PRIVATE vkexec::vkexec)
```

Configure the consumer with `-DCMAKE_PREFIX_PATH=/path/to/vkexec-prefix`. Fetched dependencies are installed alongside vkexec when their upstream projects provide install rules. Dependencies supplied externally remain external and must be discoverable by the consumer. The platform Vulkan SDK or loader development files must also be available. The source-tree and installed package expose the same `vkexec::*` target names listed in the README. The checked-in [`test/install_consumer/CMakeLists.txt`](https://github.com/janagor/vkexec/blob/main/test/install_consumer/CMakeLists.txt) is the consumer fixture; `vkexec.install_consumer` installs the built project, relocates its prefix, then configures and links independent consumers for each installed module.

## Consume the source tree

A parent CMake project can add vkexec as a subdirectory and link the same namespaced target:

```cmake
add_subdirectory(vkexec)
target_link_libraries(app PRIVATE vkexec::vkexec)
```

The [`test/subdirectory_consumer/CMakeLists.txt`](https://github.com/janagor/vkexec/blob/main/test/subdirectory_consumer/CMakeLists.txt) fixture checks this path under parent standard levels 17 and 23; vkexec still exports its own C++20 requirement without changing the parent's global standard. `vkexec.subdirectory_consumer.cxx17` and `.cxx23` run those checks in CTest.

If a parent project uses CMake `FetchContent`, declare a pinned vkexec revision and call `FetchContent_MakeAvailable(vkexec)` before linking `vkexec::vkexec`. That produces the same subdirectory integration as above. Set vkexec options before `FetchContent_MakeAvailable`; the first configure also fetches vkexec's dependencies from `Dependencies.cmake`.
