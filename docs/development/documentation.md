# Documentation ownership

`main/docs/` is the only hand-maintained documentation tree. `README.md` orients repository visitors; GitHub Pages is generated from docs and public headers. The Pages artifact is derived output, and a `gh-pages` branch must not be edited as documentation source.

| Information | Authority |
|---|---|
| C++ standard | `target_compile_features()` |
| CMake minimum | root `CMakeLists.txt` |
| Compiler floors | CI matrix |
| Public targets and build options | CMake target and option files |
| Dependency revisions | `Dependencies.cmake` |
| API signatures | installed public headers |
| Runnable examples | `examples/` |
| Architectural invariants | `docs/architecture/invariants.md` |

Build locally with `doxygen Doxyfile` followed by `sphinx-build -W --keep-going -b html docs docs/_build/html`. The docs workflow runs these steps on pull requests and deploys only from main. Substantial code snippets should correspond to compiled examples.
