# Contributing

Update code, examples, and the matching documentation together. Keep public concepts and API descriptions distinct from implementation architecture. Add a compiled example when a tutorial needs substantial code. Run the relevant build, CTest, and strict documentation build before submitting a change.

# File length limits

CI checks physical lines in C++ files, including blank lines and comments. Source files (`.cpp`, `.cc`, `.cxx`) are limited to 500 lines; headers (`.hpp`, `.h`, `.hxx`) to 600 lines; and files under `test/` or `benchmark/` to 800 lines. Generated and third-party directories are excluded.

Run `python3 scripts/check_loc.py` to check the limits locally, or `python3 scripts/check_loc.py --report` to list every checked file without failing. The checker includes tracked and new, non-ignored files. Every file must meet its category limit; there are no exceptions for existing files.
