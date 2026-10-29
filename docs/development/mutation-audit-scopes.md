# Mutation audit scopes

Mull generates mutations during compilation. Each build directory selects exactly
one source scope through `vkexec_MULL_SCOPE` (default: `ranges`). Runtime Catch2
filters select tests; they cannot isolate compiled mutation source scopes.

Register scopes after their executable targets exist and before the single
`vkexec_finalize_mutation_scope()` finalization call:

```cmake
vkexec_register_mutation_scope(
  NAME ranges
  CONFIG "${PROJECT_SOURCE_DIR}/mull.yml"
  TEST_TARGET vkexec_core_tests
  SOURCES "include/vkexec/detail/resource_ranges.hpp"
  TEST_ARGS "[ranges]" "--order" "lex" "--abort")
```

Every field is required. Names use lowercase letters, digits and underscores,
starting with a letter, and must be unique. CONFIG must name an existing YAML
file; TEST_TARGET must be an existing executable. SOURCES is a nonempty list of
exact repository-relative files, without duplicates or paths outside the root.
TEST_ARGS is a nonempty CMake argument list, passed identically to baseline
profiling and Mull. Keep arguments separate; do not pass a shell command.

Finalization validates the selected name, copies its configuration to the build
root's `mull.yml`, and instruments only `vkexec` and the selected executable,
once per target. Inactive registrations create no targets or instrumentation.
The YAML remains responsible for compile-time source filtering; validation rejects
reports containing unexpected files or missing any registered source.

Each selected scope creates `vkexec_mutation_<name>` and
`vkexec_mutation_<name>_uncovered`. They have separate raw/indexed profiles,
SQLite, JSON and text report paths under `mutation-reports/vkexec_core_<name>*`.
Both remove stale output, collect a baseline, merge indexed coverage, execute
Mull, and validate the report. Failures stop the build target. Uncovered mutants
remain separate outcomes. Global worker and timeout settings remain unchanged.

The migration preserves `vkexec_mutation_ranges`,
`vkexec_mutation_ranges_uncovered`, their report names, the original YAML,
indexed coverage, cache bypass and lexical/fail-fast Catch2 arguments. Existing
single-source Python validator callers remain compatible; positional CLI sources
now accept multiple files. Summary schema 2 and score definitions are unchanged.
Every expected file must have discovered mutants. Strict validation requires a
killed or surviving mutant somewhere in the complete report, while `parse_report`
also accepts an entirely uncovered report.

For PR 6, a registration could look like this (illustrative only):

```cmake
vkexec_register_mutation_scope(
  NAME dag
  CONFIG "${PROJECT_SOURCE_DIR}/cmake/mull/dag.yml"
  TEST_TARGET vkexec_core_tests
  SOURCES "include/vkexec/pass.hpp"
  TEST_ARGS "[dag]" "--order" "lex" "--abort")
```

Inspect the actual DAG source locations and Catch2 tags before implementing it.
Configure and compile a dedicated build directory or preset with
`-Dvkexec_MULL_SCOPE=dag`; keep the existing ranges directory independent.
This would generate `vkexec_mutation_dag` and
`vkexec_mutation_dag_uncovered`. PR 5 does not register or execute this scope,
introduce thresholds, or combine independent audit scores.

Run the focused configure checks without Mull or Vulkan:

```sh
python3 test/mutation/audit_registration/test_registration.py
```

## PR 5 verification

The saved pre-refactor report pair and SHA-256 manifest are under
`out/pr5-evidence/before/`. Three covered and three uncovered audits each
produced 47 discovered and 47 killed mutants, zero survivors, timeouts or
uncovered mutants, and 100% mutation score and evaluation coverage. Each run's
complete mutant ID/status set equals the saved report for that variant, without
requiring checkout-root normalization. Reports, profiles and logs are preserved
under `out/pr5-evidence/{ranges,ranges_uncovered}-{1,2,3}/`; the comparison is
`out/pr5-evidence/comparison.json`. These local generated artifacts are not
committed. Historical PR 4 records remain unchanged.

All three PR 3 standalone binaries passed `--verify-fixture`. The out-of-line
indexed report validated against both explicit source paths: ten discovered,
six killed, two survived and two uncovered. Its summary is
`/tmp/mull-multifile-validation.json`.

The mutation CTest checks pass, including focused configure and stub execution
checks for argument forwarding and failures in baseline execution, profile merge,
Mull execution and validation. The normal debug CTest suite completed without
failures: 348 passed and three skipped on this machine.
