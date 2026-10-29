# Development environments

On x86_64-linux and aarch64-linux, use `nix develop .#clang-mut` for Mull 22. Only `clang-mut` includes mutation tools; the default, `clang`, and `clang16` shells do not include Mull. The `clang-mut` shell exports its matching `MULL_PLUGIN` and `MULL_RUNNER` paths for the CMake arguments described in [Mutation testing](mutation-testing.md).

`flake.nix` provides the Nix development shell. The `.devcontainer` Dockerfile supplies Clang and GCC images; `devcontainer.json` mounts the checkout at `/workspaces/vkexec` when the checkout is named vkexec. Build manually with `docker build -f .devcontainer/Dockerfile --target clang -t vkexec:clang .` and run with `docker run -it -v "$(pwd)":/workspaces/vkexec -w /workspaces/vkexec vkexec:clang`. Then use the normal CMake commands.
