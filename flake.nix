{
  description = "vkexec — stdexec Vulkan compute backend";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-legacy.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      nixpkgs,
      nixpkgs-legacy,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        legacyPkgs = import nixpkgs-legacy { inherit system; };
        llvm = pkgs.llvmPackages_22;

        llvmStdenv = pkgs.overrideCC llvm.stdenv (
          llvm.stdenv.cc.override {
            bintools = llvm.bintools;
          }
        );

        llvm16 = legacyPkgs.llvmPackages_16;
        llvm16Stdenv = legacyPkgs.overrideCC llvm16.libcxxStdenv (
          llvm16.libcxxStdenv.cc.override {
            bintools = llvm16.bintools;
          }
        );

        commonPackages = p: clangTools: vulkanHeaders: with p; [
          cmake
          ninja
          gcovr
          ccache
          doxygen
          cppcheck
          graphviz
          pkg-config
          include-what-you-use
          clangTools

          vulkanHeaders
          vulkan-loader
          vulkan-validation-layers
          vulkan-tools
          shaderc
          glslang
          spirv-tools
          glfw
          (if p ? libx11 then p.libx11 else p.xorg.libX11)
          (if p ? libxrandr then p.libxrandr else p.xorg.libXrandr)
          (if p ? libxi then p.libxi else p.xorg.libXi)
          (if p ? libxcursor then p.libxcursor else p.xorg.libXcursor)
          (if p ? libxinerama then p.libxinerama else p.xorg.libXinerama)
          libxkbcommon
          wayland
          wayland-protocols
          wayland-scanner
          libffi
          libdecor
          kdePackages.extra-cmake-modules
        ];

        runtimeLibs = with pkgs; [
          vulkan-loader
          libx11
          libxrandr
          libxi
          libxcursor
          libxinerama
          libxkbcommon
          wayland
          libffi
          libdecor
        ];

        shellEnv = {
          VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
          # GLFW 3.4 dlopens X11/Wayland at runtime; expose them in the nix shell.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
        };

        clangShell = pkgs.mkShell.override { stdenv = llvmStdenv; } (
          shellEnv
          // {
            packages = commonPackages pkgs llvm.clang-tools pkgs.vulkan-headers;
          }
        );

        gccShell = pkgs.mkShell.override { stdenv = pkgs.gcc16Stdenv; } (
          shellEnv
          // {
            packages = commonPackages pkgs llvm.clang-tools pkgs.vulkan-headers ++ [ pkgs.mold ];
            # Host environments (e.g. NixOS /usr/lib → GCC 15) inject
            # NIX_CFLAGS_LINK=-L/usr/lib, which shadows this toolchain's libstdc++
            # and breaks C++20 atomic wait/notify link symbols.
            shellHook = ''
              export NIX_CFLAGS_LINK="-L${pkgs.lib.getLib pkgs.gcc16Stdenv.cc.cc}/lib ''${NIX_CFLAGS_LINK:-}"
            '';
          }
        );

        gcc12Shell = legacyPkgs.mkShell.override { stdenv = legacyPkgs.gcc12Stdenv; } (
          shellEnv // {
            packages = commonPackages legacyPkgs llvm.clang-tools pkgs.vulkan-headers ++ [ legacyPkgs.mold ];
            shellHook = ''
              export NIX_CFLAGS_LINK="-Wl,-rpath,${legacyPkgs.lib.getLib legacyPkgs.gcc12Stdenv.cc.cc}/lib ''${NIX_CFLAGS_LINK:-}"
            '';
          }
        );

        clang16Shell = legacyPkgs.mkShell.override {
          stdenv = llvm16Stdenv;
        } (
          shellEnv // {
            packages = commonPackages legacyPkgs llvm.clang-tools pkgs.vulkan-headers;
          }
        );
      in
      {
        packages.sample-shader-compiler = pkgs.shaderc;

        devShells = {
          default = clangShell;
          gcc = gccShell;
          clang = clangShell;
        } // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          gcc12 = gcc12Shell;
          clang16 = clang16Shell;
        };
      }
    );
}
