{
  description = "vkexec — stdexec Vulkan compute backend";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        llvm = pkgs.llvmPackages_22;

        llvmStdenv = pkgs.overrideCC llvm.stdenv (
          llvm.stdenv.cc.override {
            bintools = llvm.bintools;
          }
        );

        commonPackages = with pkgs; [
          cmake
          ninja
          gcovr
          ccache
          doxygen
          cppcheck
          graphviz
          pkg-config
          include-what-you-use
          llvm.clang-tools

          vulkan-headers
          vulkan-loader
          vulkan-validation-layers
          vulkan-tools
          shaderc
          glslang
          spirv-tools
          glfw
          libx11
          libxrandr
          libxi
          libxcursor
          libxinerama
          libxkbcommon
        ];

        clangShell = pkgs.mkShell.override { stdenv = llvmStdenv; } {
          packages = commonPackages;
          VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
        };

        gccShell = pkgs.mkShell.override { stdenv = pkgs.gcc16Stdenv; } {
          packages = commonPackages ++ [ pkgs.mold ];
          VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
        };
      in
      {
        devShells = {
          default = clangShell;
          gcc = gccShell;
          clang = clangShell;
        };
      }
    );
}
