# Compiler support

Public targets require C++20 through `target_compile_features(... PUBLIC cxx_std_20)`. The current CI floor lanes cover GCC 12, Clang 16, and MSVC 19.43 in Debug and Release. See `.github/workflows/ci.yml` for the current platform matrix.
