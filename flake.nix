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
          pre-commit

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

        clangMutShell = pkgs.mkShell.override { stdenv = llvmStdenv; } (
          shellEnv // {
            packages = commonPackages pkgs llvm.clang-tools pkgs.vulkan-headers ++ [ mull22 pkgs.python3 ];
            # Mull uses -O0, while Nix fortify requires optimization.
            hardeningDisable = [ "fortify" ];
            MULL_PLUGIN = "${mull22}/lib/mull-ir-frontend-22";
            MULL_RUNNER = "${mull22}/bin/mull-runner-22";
            # Mull's scanner does not follow the executable's ELF RUNPATH.
            MULL_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.vulkan-loader
              pkgs.glibc
              pkgs.stdenv.cc.cc
              pkgs.stdenv.cc.cc.libgcc
            ];
          }
        );

        mullVersion = "0.35.0";
        mullPlatforms = {
          x86_64-linux = "amd64";
          aarch64-linux = "aarch64";
        };
        mullSupported = builtins.hasAttr system mullPlatforms;
        mkMull = { toolchain, major, llvmVersion, distribution, release, distributionVersion, hashes }:
          assert toolchain.llvm.version == llvmVersion;
          pkgs.stdenvNoCC.mkDerivation {
            pname = "mull-${major}-bin";
            version = mullVersion;
            src = pkgs.fetchurl {
              url = "https://dl.cloudsmith.io/public/mull-project/mull-stable/deb/${distribution}/pool/${release}/main/M/Mu/mull-${major}_${mullVersion}/Mull-${major}-${mullVersion}-LLVM-${llvmVersion}-${distribution}-${mullPlatforms.${system}}-${distributionVersion}.deb";
              sha256 = hashes.${system};
            };
            nativeBuildInputs = [ pkgs.dpkg pkgs.autoPatchelfHook ];
            buildInputs = [
              (pkgs.lib.getLib toolchain.llvm)
              (pkgs.lib.getLib toolchain.clang-unwrapped)
              (pkgs.lib.getLib pkgs.stdenv.cc.cc)
            ];
            dontUnpack = true;
            dontConfigure = true;
            dontBuild = true;
            installPhase = ''
              runHook preInstall
              mkdir -p "$out" "$TMPDIR/mull-extracted"
              dpkg-deb -x "$src" "$TMPDIR/mull-extracted"
              cp -a "$TMPDIR/mull-extracted/usr/." "$out/"
              runHook postInstall
            '';
            meta = {
              description = "Mull mutation testing tools and IR frontend for LLVM ${llvmVersion}";
              homepage = "https://mull-project.com/";
              license = pkgs.lib.licenses.asl20;
              platforms = builtins.attrNames mullPlatforms;
              sourceProvenance = [ pkgs.lib.sourceTypes.binaryNativeCode ];
            };
          };

        mull22 = mkMull {
          toolchain = llvm;
          major = "22";
          llvmVersion = "22.1.8";
          distribution = "debian";
          release = "trixie";
          distributionVersion = "13";
          hashes = {
            x86_64-linux = "5896b2a33691ddd406054607662d7bfc00b44d0608db2a9d01df549820eec857";
            aarch64-linux = "5eb8a7faf3d83a0422654f6098db9048d245e21a88c425c0d9b457e3157429d7";
          };
        };
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
        packages = {
          sample-shader-compiler = pkgs.shaderc;
        } // pkgs.lib.optionalAttrs mullSupported {
          mull = mull22;
        };

        devShells = {
          default = clangShell;
          gcc = gccShell;
          clang = clangShell;
        } // pkgs.lib.optionalAttrs mullSupported {
          clang-mut = clangMutShell;
        } // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          gcc12 = gcc12Shell;
          clang16 = clang16Shell;
        };
      }
    );
}
