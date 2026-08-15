## Docker Instructions

The `.devcontainer` image provides a full C++ build environment on Ubuntu 24.04 (`noble`):

**cmake**, **ninja**, **gcovr**, **ccache**, **doxygen**, **cppcheck**, **graphviz**, **pkg-config**, **include-what-you-use**, **clang-tidy**, **GCC 16**, **LLVM/Clang 22**, and **mold**.

Two build targets set the default compiler:

| Target | Default `CC`/`CXX` | Typical CMake preset |
| --- | --- | --- |
| `clang` (default) | `clang` / `clang++` | `unixlike-clang-release` |
| `gcc` | `gcc` / `g++` | `unixlike-gcc-release` |

Optional build args: `VARIANT` (default `noble`), `GCC_VER` (default `16`), `LLVM_VER` (default `22`).

### Build the image

From the repository root:

```bash
docker build -f ./.devcontainer/Dockerfile --target clang --tag=cmake_template:clang .
docker build -f ./.devcontainer/Dockerfile --target gcc --tag=cmake_template:gcc .
```

Omitting `--target` builds the `clang` image (last stage in the Dockerfile).

### Run the container

Dev Containers / Cursor bind-mount the workspace automatically via [devcontainer.json](.devcontainer/devcontainer.json).

For a manual shell, mount your checkout:

```bash
docker run -it \
  -v "$(pwd)":/workspaces/cmake_template \
  -w /workspaces/cmake_template \
  cmake_template:clang
```

Use `cmake_template:gcc` for the GCC image.

### Configure and build

Use a fresh build directory if you previously configured on the host and see cache path errors:

```bash
rm -rf out/build/unixlike-clang-release   # or unixlike-gcc-release
```

**Clang** (`cmake_template:clang`):

```bash
cmake --preset unixlike-clang-release
cmake --build out/build/unixlike-clang-release -j"$(nproc)"
```

**GCC** (`cmake_template:gcc`):

```bash
cmake --preset unixlike-gcc-release
cmake --build out/build/unixlike-gcc-release -j"$(nproc)"
```

**Tests:**

```bash
ctest --preset test-unixlike-clang-release
# or
ctest --preset test-unixlike-gcc-release
```

The `gcc` target includes **mold** for linker presets that request it. Both targets include **clang-tidy** regardless of which compiler you build with.
