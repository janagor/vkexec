"""Orchestrate existing public CMake audits, retaining evidence on failure."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_mutation_baselines import SCOPES, compare, fingerprint, read, require, sha, validate_pair


def run(command, log):
    with log.open('w') as stream:
        stream.write('Command: ' + repr(command) + '\n')
        stream.flush()
        return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode


def version(command, pattern, expected):
    text = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    require(re.search(pattern + re.escape(expected) + r'(?:\D|$)', text) is not None, f'Unexpected version: {text}')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scope', choices=['all', *SCOPES], default='all', nargs='?')
    parser.add_argument('--policy', choices=['report', 'regression'], default='report')
    parser.add_argument('--artifacts', type=Path, default=Path('out/mutation-ci-artifacts'))
    parser.add_argument('--measure-only', action='store_true', help='Collect validated evidence for explicit baseline creation; no comparison')
    args = parser.parse_args()
    root = Path.cwd().resolve()
    build = root / 'out/build/unixlike-clang-mut'
    reports = build / 'mutation-reports'
    artifacts = args.artifacts.resolve()
    require(not artifacts.exists(), f'Artifact destination already exists: {artifacts}; use a fresh directory')
    artifacts.mkdir(parents=True)
    diagnostics = artifacts / 'diagnostics'
    diagnostics.mkdir()
    selected = list(SCOPES) if args.scope == 'all' else [args.scope]
    git = lambda *a: subprocess.check_output(['git', *a], text=True).strip()
    manifest = {'schema_version': 1, 'revision': git('rev-parse', 'HEAD'), 'source_clean': not bool(git('status', '--porcelain')), 'policy': args.policy, 'measure_only': args.measure_only, 'scopes': {}, 'errors': []}
    failed = False
    summary = ['# Mutation audit', '', '| Scope | Killed | Survived | Uncovered | Regressions | Comparable | Result |', '|---|---:|---:|---:|---:|---|---|']
    try:
        for binary in ('cmake', 'ctest', 'clang++', 'llvm-profdata', 'mull-runner-22', 'ninja'):
            require(shutil.which(binary), f'Missing binary: {binary}')
        plugin = os.environ.get('MULL_PLUGIN', '')
        runner = os.environ.get('MULL_RUNNER', '')
        require(plugin.startswith('/nix/store/') and Path(plugin).is_file(), 'Missing Nix Mull plugin')
        require(runner.startswith('/nix/store/') and Path(runner).is_file(), 'Missing Nix Mull runner')
        require(os.environ.get('MULL_LIBRARY_PATH'), 'Missing Nix Mull library paths')
        versions = {
            'clang': version(['clang++', '--version'], r'clang version ', '22.1.8'),
            'llvm': version(['llvm-profdata', '--version'], r'LLVM version ', '22.1.8'),
            'mull': version([runner, '--version'], r'mull-runner ', '0.35.0'),
        }
        (diagnostics / 'versions.json').write_text(json.dumps(versions, indent=2) + '\n')
        os.environ['CMAKE_BUILD_PARALLEL_LEVEL'] = '2'
        command = ['cmake', '--preset', 'unixlike-clang-mut', '-Dvkexec_MULL_WORKERS=1', '-Dvkexec_MULL_TIMEOUT_MS=10000', '-Dvkexec_MULL_MINIMUM_TIMEOUT_MS=1000']
        require(run(command, diagnostics / 'configure.log') == 0, 'CMake configure failed')
        cache = (build / 'CMakeCache.txt').read_text()
        options = {}
        for line in cache.splitlines():
            match = re.match(r'((?:vkexec_|CMAKE_BUILD_TYPE|CMAKE_CXX_FLAGS|CMAKE_EXE_LINKER_FLAGS|BUILD_TESTING)[^:]*):(?:BOOL|STRING)=(.*)', line)
            if match and not any(x in match[1] for x in ('_DIR', '_PATH', 'MULL_', 'MUTATION_TARGETS')):
                options[match[1]] = match[2]
        manifest['execution'] = {'toolchain': {'clang': '22.1.8', 'llvm': '22.1.8', 'mull': '0.35.0'}, 'build_options': options, 'timeout_policy': {'warmup_ms': 10000, 'minimum_ms': 1000, 'workers': 1}}
        code = run(['ctest', '--test-dir', str(build), '-R', r'^vkexec\.mull_(report_validator|coverage_diagnostics|audit_registration)$', '--no-tests=error', '--output-on-failure'], diagnostics / 'ctest.log')
        require(code == 0, 'Mutation infrastructure CTests failed')
        for scope in selected:
            destination = artifacts / scope
            destination.mkdir()
            record = {'test_args': [SCOPES[scope][2], '--order', 'lex', '--abort'], 'sources': SCOPES[scope][1]}
            manifest['scopes'][scope] = record
            try:
                for name, suffix in [('covered', ''), ('inclusive', '_uncovered')]:
                    stem = 'vkexec_core_' + scope + suffix
                    # Remove this scope's previous outputs so failures cannot reuse stale evidence.
                    for extension in ('json', 'sqlite', 'txt', 'profraw', 'profdata'):
                        (reports / f'{stem}.{extension}').unlink(missing_ok=True)
                    record[name + '_returncode'] = run(['cmake', '--build', str(build), '--target', 'vkexec_mutation_' + scope + suffix], destination / f'{name}.log')
                    for extension in ('json', 'sqlite', 'txt', 'profraw', 'profdata'):
                        source = reports / f'{stem}.{extension}'
                        if source.is_file():
                            shutil.copy2(source, destination / f'{name}.{extension}')
                require(record['covered_returncode'] == record['inclusive_returncode'] == 0, 'Audit target failed (see per-target logs)')
                for name in ('covered', 'inclusive'):
                    for extension in ('json', 'sqlite', 'txt', 'profraw', 'profdata'):
                        require((destination / f'{name}.{extension}').is_file(), f'Missing {name}.{extension}')
                    record[name + '_sha256'] = sha(destination / f'{name}.json')
                covered, inclusive = read(destination / 'covered.json'), read(destination / 'inclusive.json')
                validate_pair(covered, inclusive, scope, root)
                record['fingerprint'] = fingerprint(root, scope, manifest['execution'])
                if args.measure_only:
                    result = {'status': 'measured', 'counts': {s: inclusive[s] for s in ('killed', 'survived', 'not_covered')}, 'regressions': 0, 'comparable': False, 'changes': [], 'exit_code': 0}
                else:
                    result = compare(scope, read(root / 'docs/development/mutation-ci-baselines' / (scope + '.json')), covered, inclusive, root, record['fingerprint'], args.policy)
                record['comparison'] = result
                (destination / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
                failed |= result['exit_code'] != 0
                counts = result['counts']
                summary.append(f"| {scope} | {counts.get('killed', 0)} | {counts.get('survived', 0)} | {counts.get('not_covered', 0)} | {result['regressions']} | {result['comparable']} | {result['status']} |")
                details = [r for r in result['changes'] if r['classification'] != 'unchanged']
                record['summary_details'] = details
            except (ValueError, OSError, KeyError, TypeError) as error:
                failed = True
                record['error'] = str(error)
                (destination / 'comparison.json').write_text(json.dumps({'scope': scope, 'status': 'invalid', 'error': str(error), 'exit_code': 1}, indent=2) + '\n')
                summary.append(f'| {scope} | — | — | — | — | — | invalid: {error} |')
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        failed = True
        manifest['errors'].append(str(error))
    finally:
        for scope, record in manifest['scopes'].items():
            summary += ['', f'## {scope}', '', f'Artifacts: `{scope}/` (reports, profiles, target logs and comparison).']
            comparison = record.get('comparison', {})
            summary += ['', 'Fingerprint mismatches: ' + ', '.join(comparison.get('fingerprint_mismatches', []))]
            for row in record.pop('summary_details', []):
                summary.append(f"- `{row['id']}`: {row['previous']} → {row['current']} ({row['classification']})")
        summary += ['', *manifest['errors']]
        for path in [build / 'CMakeCache.txt', build / 'Testing/Temporary/LastTest.log', root / 'flake.lock']:
            if path.is_file():
                shutil.copy2(path, diagnostics / path.name)
        manifest['source_clean'] = manifest['source_clean'] and not bool(git('status', '--porcelain')) and manifest['revision'] == git('rev-parse', 'HEAD')
        manifest['failed'] = bool(failed)
        (artifacts / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        text = '\n'.join(summary) + '\n'
        (artifacts / 'summary.md').write_text(text)
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
                stream.write(text)
        print(text)
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
