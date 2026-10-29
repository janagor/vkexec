# Resource-range uncovered mutation investigation (PR 2)

Measured 2026-10-29 against the existing PR 1 SQLite reports, using the same
instrumented executable and `[ranges]` filter. LLVM coverage tools are 22.1.8.
The source implementation and tests match revision
`3b181e016055f99a012ede9330629a86073bf21c`; this investigation changes only
reporting, diagnostic fixtures and documentation. The PR 1 baseline remains a
historical measurement and is not relabeled as a new clean baseline.

All **43 uncovered mutants** map to executed LLVM code regions. Four other
mutants are killed and also map to executed regions. Evaluated/discovered is
**4/47 (8.51%)**, independently of the 100% mutation score over four mutants.
The original tests again passed 72,918 assertions in 53 cases.

The [location evidence](mutation-coverage-investigation.json) classifies every
reported mutant, retaining mangled function names and execution counts for each
LLVM function instantiation at the mutation point. Each count comes from the
innermost containing code region, with exclusive end coordinates; gap and
expansion regions are excluded. This is stronger evidence than function-entry
or whole-line coverage, but cannot establish which binary instance Mull chose.

| Function | Source lines | Uncovered mutants | Classification |
| --- | --- | ---: | --- |
| normalize(buffer_use) | 30–35 | 8 | Source executed; attribution discrepancy |
| overlaps(interval), both integer instantiations | 40 | 4 | Source executed; attribution discrepancy |
| split | 67 | 4 | Source executed; attribution discrepancy |
| ambiguous_open_remainder, both integer instantiations | 75 | 2 | Source executed; attribution discrepancy |
| adjacent, both integer instantiations | 85 | 2 | Source executed; attribution discrepancy |
| region_size | 88 | 1 | Source executed; attribution discrepancy |
| ambiguous_open_uncovered(image_region) | 104 | 1 | Source executed; attribution discrepancy |
| normalize_subresources | 113–118 | 8 | Source executed; attribution discrepancy |
| normalize(image range) | 124–125 | 3 | Source executed; attribution discrepancy |
| overlaps(image_region) | 135 | 1 | Source executed; attribution discrepancy |
| intersection(image_region) | 142 | 1 | Source executed; attribution discrepancy |
| subresource_range | 150–152 | 2 | Source executed; attribution discrepancy |
| subtract | 161–173 | 5 | Source executed; attribution discrepancy |
| merge_compatible | 194 | 1 | Source executed; attribution discrepancy |

The build uses a static core library (`BUILD_SHARED_LIBS=OFF`), so an omitted
shared-library coverage object does not explain these results. `llvm-nm` shows
weak symbols for normalize, overlaps, split and contains in the executable.
LLVM export retains both uint32 and VkDeviceSize template instantiations for
several helpers. These observations make inline instance selection or coverage
mapping candidates for further investigation; they do **not** establish an exact
Mull defect. No genuinely unexecuted mutation location was identified by this
source-region comparison. No range assertions, production code, mutant exclusions,
DAG scope or score gates were added.

Before relying on uncovered status for PR 3, reduce this discrepancy to a small
multi-translation-unit inline example and inspect Mull's binary-instance coverage
selection. Source coverage validation is complete here; binary-instance root cause
and a toolchain fix remain unresolved. Do not interpret these 43 classifications
as kills or as proof that existing assertions detect those mutants.

Reproduce after building the mutation executable (from the `clang-mut` shell):

```sh
LLVM_PROFILE_FILE=/tmp/vkexec-ranges.profraw \
  out/build/unixlike-clang-mut/test/vkexec_core_tests '[ranges]'
llvm-profdata merge -sparse /tmp/vkexec-ranges.profraw -o /tmp/vkexec-ranges.profdata
llvm-cov export out/build/unixlike-clang-mut/test/vkexec_core_tests \
  -instr-profile=/tmp/vkexec-ranges.profdata \
  include/vkexec/detail/resource_ranges.hpp > /tmp/vkexec-ranges-coverage.json
python3 scripts/diagnose_mull_coverage.py \
  out/build/unixlike-clang-mut/mutation-reports/vkexec_core_ranges_uncovered.sqlite \
  include/vkexec/detail/resource_ranges.hpp /tmp/vkexec-ranges-coverage.json \
  --json-output /tmp/mutation-coverage-investigation.json
python3 scripts/test_validate_mull_report.py
python3 scripts/test_diagnose_mull_coverage.py
```

Use a fresh profile path for each run. Regenerate the SQLite report when changing
the executable; comparing stale reports with a different executable is invalid.
For shared builds supply the instrumented library with `llvm-cov -object` as well.
The diagnostic tool consumes LLVM's export format; missing source regions remain
explicitly unclassified (`no_source_region`), rather than being called test gaps.

The diagnostic CLI uses `parse_report()` to check SQLite structure, source scope,
locations, identifiers and supported statuses without requiring a killed or
surviving mutant. Entirely uncovered reports can therefore be investigated.
`validate()` retains that requirement for the CMake mutation audits. Empty reports,
missing reports and malformed or unsupported data remain errors in both paths.

Generated diagnostics preserve Mull's raw `mutant_id` for correlation with the
original SQLite row. The committed artifact contains 47 distinct IDs with the
checkout root removed from their embedded source paths. Local SQLite reports and
generated JSON retain the raw values. The observed IDs encode the mutator, source
path and source span; they do not expose a translation unit or binary address.
Keeping them does not resolve binary-instance identity.

Before committing a regenerated artifact, normalize this known path prefix from
the repository root (after the reproduction commands above):

```sh
python3 - <<'PY'
import json
from pathlib import Path

source = Path('/tmp/mutation-coverage-investigation.json')
artifact = Path('docs/development/mutation-coverage-investigation.json')
report = json.loads(source.read_text())
prefix = Path.cwd().as_posix() + '/'
for location in report['locations']:
    location['mutant_id'] = location['mutant_id'].replace(prefix, '')
artifact.write_text(json.dumps(report, indent=2) + '\n')
PY
```
