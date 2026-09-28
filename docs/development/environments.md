# Development environments

`flake.nix` provides the Nix development shell. The `.devcontainer` Dockerfile supplies Clang and GCC images; `devcontainer.json` mounts the checkout at `/workspaces/vkexec` when the checkout is named vkexec. Build manually with `docker build -f .devcontainer/Dockerfile --target clang -t vkexec:clang .` and run with `docker run -it -v "$(pwd)":/workspaces/vkexec -w /workspaces/vkexec vkexec:clang`. Then use the normal CMake commands.
