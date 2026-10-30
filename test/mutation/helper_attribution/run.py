"""Build and measure the standalone pinned helper attribution experiment.

Run from the repository root in nix develop .#clang-mut. Retains all raw evidence.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build', type=Path, default=Path('out/build/mull-helper-fixture'))
parser.add_argument('--artifacts', type=Path, default=Path('out/pr10-evidence/fixture'))
args = parser.parse_args()
build, artifacts = args.build.resolve(), args.artifacts.resolve()
artifacts.mkdir(parents=True, exist_ok=True)

def run(command, log):
    with log.open('w') as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, check=True)

run(['cmake', '-S', 'test/mutation/helper_attribution', '-B', build, '-G', 'Ninja',
     '-DCMAKE_CXX_COMPILER=clang++'], artifacts / 'configure.log')
run(['cmake', '--build', build, '--parallel', '2'], artifacts / 'build.log')
summary = {}
for target in ('single', 'multi', 'single_ordinary', 'multi_ordinary', 'out_of_line', 'single_if'):
    destination = artifacts / target
    run([sys.executable, 'scripts/compare_mull_coverage.py', build / target,
         build / 'mull.yml', destination], artifacts / (target + '.log'))
    run(['llvm-cov', 'export', build / target,
         '-instr-profile=' + str(destination / 'manual.profdata')], destination / 'coverage-export.json')
    result = json.loads((destination / 'comparison.json').read_text())
    # Freeze the pinned reproducer's exact indexed population and statuses.
    # A future Mull correction must fail this check for explicit review.
    expected = {
        'validity.hpp:41:73:41:75': 8,  # Emitted, deliberately unexecuted control.
        'validity.hpp:48:50:48:52': 1,  # Executed runtime-oracle comparisons.
        'validity.hpp:49:63:49:65': 1,
    }
    if target == 'single_if':
        expected.update({
            'validity.hpp:23:16:23:18': 1,
            'validity.hpp:23:51:23:53': 1,
            'validity.hpp:24:16:24:18': 1,
            'validity.hpp:24:64:24:66': 1,
        })
    elif target == 'out_of_line':
        expected.update({'out_of_line.cpp:6:21:6:23': 8, 'out_of_line.cpp:8:21:8:23': 8})
    else:
        expected.update({'validity.hpp:29:21:29:23': 8, 'validity.hpp:31:21:31:23': 8})
    expected_rows = sorted((f'cxx_eq_to_ne:test/mutation/helper_attribution/{location}', status)
                           for location, status in expected.items())
    # Remove only this checkout's absolute source prefix; retain full spans and
    # mutator names so unexpected sources, duplicate IDs and new mutants fail.
    prefix = f'cxx_eq_to_ne:{Path.cwd().resolve()}/'
    actual_rows = sorted((identifier.replace(prefix, 'cxx_eq_to_ne:', 1)
                          if identifier.startswith(prefix) else identifier, status)
                         for identifier, status in result['modes']['indexed']['mutants'])
    if actual_rows != expected_rows:
        raise RuntimeError(f'{target}: indexed mutant population/status mismatch; '
                           f'expected {expected_rows!r}, got {actual_rows!r}')
    summary[target] = result
(artifacts / 'matrix.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({name: {mode: data['counts'] for mode, data in result['modes'].items()}
                  for name, result in summary.items()}, indent=2))
