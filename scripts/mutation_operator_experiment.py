"""Isolated OR-operator experiment; never writes canonical baselines.

Run in nix develop .#clang-mut from a clean source checkout. Evidence destinations
must be fresh. The wrapper itself and its configurations may live outside that
checkout; their hashes are retained alongside the measured source revision.
"""
import argparse
from collections import Counter
import json
import os
import re
from pathlib import Path
import sqlite3
from contextlib import closing
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_mutation_baselines import normalize, read, require, sha, sources, fingerprint
from validate_mull_report import parse_report
from diagnose_mull_coverage import diagnose

NEW = ('cxx_or_to_and', 'cxx_or_assign_to_and_assign')
OLD = ('cxx_comparison', 'cxx_boundary')
CONFIG = 'cmake/mull/resource_helpers.yml'
CLASSIFICATIONS = {'test weakness', 'known attribution limitation', 'potential equivalence', 'unresolved'}
ROOT = Path(__file__).resolve().parents[1]


def configuration(variant):
    operators = (*OLD, *NEW) if variant == 'combined' else NEW
    paths = ''.join("  - '(^|.*/)" + p.replace('.', r'\.') + "$'\n" for p in sources('resource_helpers'))
    return 'mutators:\n' + ''.join('  - ' + op + '\n' for op in operators) + '\nincludePaths:\n' + paths


def validate_configuration(text, variant):
    # An intentionally narrow schema: no YAML aliases, extra keys, broad groups,
    # duplicate operators, or regex allowlist expansion can pass unnoticed.
    require(variant in ('or', 'combined'), 'Unknown experiment variant')
    require(text == configuration(variant), 'Unexpected operators or source allowlist')


def execute(command, cwd, log, env=None):
    with log.open('w') as stream:
        if log.suffix == '.json':
            log.with_suffix('.command.txt').write_text(repr(list(map(str, command))) + '\n')
        else:
            stream.write(repr(list(map(str, command))) + '\n')
            stream.flush()
        subprocess.run(list(map(str, command)), cwd=cwd, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def check_overlay(worktree):
    require(git(worktree, 'diff', '--name-only', 'HEAD') == CONFIG, 'Unexpected tracked-file modification')
    require(not git(worktree, 'ls-files', '--others', '--exclude-standard'), 'Unexpected untracked files')
    require(not git(worktree, 'diff', '--cached', '--name-only'), 'Unexpected staged changes')


def tool_versions():
    result = {}
    for name, command, marker in (
        ('clang', ['clang++', '--version'], 'clang version 22.1.8'),
        ('llvm', ['llvm-profdata', '--version'], 'LLVM version 22.1.8'),
        ('mull', [os.environ.get('MULL_RUNNER', 'mull-runner-22'), '--version'], 'mull-runner 0.35.0'),
    ):
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        require(re.search(re.escape(marker) + r'(?:\s|$)', output) is not None, f'Unexpected {name} version: {output}')
        result[name] = output
    require(Path(os.environ.get('MULL_PLUGIN', '')).is_file(), 'Missing Mull plugin')
    return result


def verify_operators(directory):
    directory.mkdir()
    fixture = directory / 'fixture.cpp'
    fixture.write_text('''__attribute__((noinline)) unsigned combine(unsigned a, unsigned b) { return a | b; }
__attribute__((noinline)) unsigned accumulate(unsigned a, unsigned b) { a |= b; return a; }
int main(int argc, char**) { volatile unsigned a = argc; volatile unsigned b = 128;
return combine(a, b) == a + b && accumulate(a, b) == a + b ? 0 : 1; }
''')
    config = directory / 'mull.yml'
    config.write_text('mutators:\n' + ''.join('  - ' + op + '\n' for op in NEW) +
                      "includePaths:\n  - '(^|.*/)fixture\\.cpp$'\n")
    env = dict(os.environ, MULL_CONFIG=str(config), LLVM_PROFILE_FILE=str(directory / 'fixture.profraw'))
    binary = directory / 'fixture'
    execute(['clang++', '-g', '-O0', '-fprofile-instr-generate', '-fcoverage-mapping',
             '-grecord-command-line', '-fpass-plugin=' + os.environ['MULL_PLUGIN'], fixture, '-o', binary], directory, directory / 'compile.log', env)
    execute([binary], directory, directory / 'original.log', env)
    libraries = [item for p in os.environ.get('MULL_LIBRARY_PATH', '').split(':') if p for item in ('--ld-search-path', p)]
    execute([os.environ.get('MULL_RUNNER', 'mull-runner-22'), '--allow-surviving', '--include-not-covered',
             '--reporters', 'SQLite', '--report-dir', directory, '--report-name', 'operators',
             *libraries, binary], directory, directory / 'mull.log', env)
    with closing(sqlite3.connect((directory / 'operators.sqlite').as_uri() + '?mode=ro', uri=True)) as db:
        rows = db.execute('SELECT mutator, execution_status FROM mutant').fetchall()
    require(Counter(op for op, _ in rows) == Counter(NEW), f'Operators not emitted exactly once: {rows}')
    require(all(status == 1 for _, status in rows), f'Fixture mutants must be killed: {rows}')
    return rows


def load_pair(directory, worktree):
    reports = []
    allowed = sources('resource_helpers')
    for mode in ('covered', 'inclusive'):
        # Covered output may legitimately omit an allowlisted file if every
        # emitted mutant in that file is uncovered. Inclusive must contain both.
        expected = allowed
        if mode == 'covered':
            database = directory / 'covered.sqlite'
            require(database.is_file(), 'Missing covered SQLite report')
            with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
                files = set()
                for filename, source_dir in db.execute('SELECT filename, directory FROM mutant'):
                    path = Path(filename)
                    if not path.is_absolute():
                        require(Path(source_dir).is_absolute(), 'Invalid source directory')
                        path = Path(source_dir) / path
                    relative = path.resolve().relative_to(worktree.resolve()).as_posix()
                    require(relative in allowed, 'Unexpected source file')
                    files.add(relative)
                expected = sorted(files)
        raw = parse_report(directory / (mode + '.sqlite'), [worktree / p for p in expected], worktree)
        summary_path = directory / (mode + '.json')
        if summary_path.exists():
            require(raw == read(summary_path), f'{mode}: SQLite/JSON contradiction')
        else:
            require(mode == 'covered', 'Missing inclusive JSON report')
        rows = {}
        for row in raw['locations']:
            identifier = normalize(row, worktree)
            require(identifier not in rows, 'Duplicate normalized ID')
            rows[identifier] = {**row, 'id': identifier}
            rows[identifier].pop('mutant_id')
        (directory / ('validated-' + mode + '.json')).write_text(json.dumps(raw, indent=2) + '\n')
        reports.append(rows)
    covered, inclusive = reports
    require(all(r['status'] != 'not_covered' for r in covered.values()), 'Covered contains uncovered mutant')
    require(covered == {i: r for i, r in inclusive.items() if r['status'] != 'not_covered'},
            'Covered/inclusive contradiction')
    require(any(r['status'] in ('killed', 'survived') for r in covered.values()), 'No meaningful evaluated mutants')
    return inclusive


def compatibility(rows, baseline):
    before = {r['id']: r['status'] for r in baseline['mutants']}
    require(len(before) == len(baseline['mutants']) == 56, 'Expected 56 distinct canonical mutants')
    original = {i: r['status'] for i, r in rows.items() if r['mutator'] not in NEW}
    changes = [{'id': i, 'previous': before.get(i), 'current': original.get(i)}
               for i in sorted(before.keys() | original.keys()) if before.get(i) != original.get(i)]
    return {'expected': len(before), 'observed': len(original), 'changes': changes, 'compatible': not changes}


def classify(row, classification='unresolved', evidence=None, proposed_assertion=None):
    require(classification in CLASSIFICATIONS, 'Unexpected outcome classification')
    return {**row, 'function': None, 'classification': classification,
            'evidence': evidence or [], 'proposed_assertion': proposed_assertion}


def experiment(root, variant, revision, artifacts, command=execute, verify=verify_operators, versions=tool_versions):
    root, artifacts = root.resolve(), artifacts.resolve()
    require(not git(root, 'status', '--porcelain'), 'Measurement requires a clean committed source checkout')
    commit = git(root, 'rev-parse', '--verify', '--end-of-options', revision + '^{commit}')
    config = ROOT / 'cmake/mull/experiments' / ('resource_helpers_or' + ('_combined' if variant == 'combined' else '') + '.yml')
    text = config.read_text()
    validate_configuration(text, variant)
    require(not artifacts.exists(), 'Artifact destination already exists')
    artifacts.mkdir(parents=True)
    manifest = {'schema_version': 1, 'kind': 'experimental snapshot', 'revision': commit,
                'variant': variant, 'source_clean_before_overlay': True, 'source_clean': False,
                'intentional_override': CONFIG, 'configuration_sha256': sha(config),
                'wrapper_sha256': sha(Path(__file__)), 'test_args': ['[resource-helpers]', '--order', 'lex', '--abort'],
                'failed': True, 'evidence': str(artifacts)}
    (artifacts / 'experiment.yml').write_text(text)
    parent = Path(tempfile.mkdtemp(prefix='vkexec-operators-'))
    worktree = parent / 'source'
    added = False
    started = time.monotonic()
    try:
        manifest['toolchain_versions'] = versions()
        manifest['operator_verification'] = verify(artifacts / 'operator-fixture')
        manifest['source_configuration'] = str(config)
        command(['git', 'worktree', 'add', '--detach', str(worktree), commit], root, artifacts / 'worktree.log')
        added = True
        manifest['original_configuration_sha256'] = sha(worktree / CONFIG)
        (worktree / CONFIG).write_text(text)
        check_overlay(worktree)
        audit_error = None
        try:
            command([sys.executable, 'scripts/ci/run_mutation_audits.py', 'resource_helpers', '--measure-only',
                     '--artifacts', str(artifacts / 'audit')], worktree, artifacts / 'audit.log')
        except subprocess.CalledProcessError as error:
            audit_error = error.returncode
        check_overlay(worktree)
        audit = read(artifacts / 'audit/manifest.json')
        require(audit['revision'] == commit and audit['source_clean'] is False, 'Invalid audit provenance')
        if audit_error or audit['failed']:
            record = audit['scopes']['resource_helpers']
            log = (artifacts / 'audit/resource_helpers/covered.log').read_text()
            require(audit_error == 1 and not audit['errors'] and record['covered_returncode'] != 0
                    and record['inclusive_returncode'] == 0
                    and 'Mull report validation failed: Expected sources have no mutants:' in log
                    and not (artifacts / 'audit/resource_helpers/covered.json').exists(),
                    'Unexpected canonical runner failure')
            manifest['canonical_runner_limitation'] = 'Covered population omits an allowlisted file; independently validated by wrapper'
            manifest['canonical_runner_returncode'] = audit_error
        rows = load_pair(artifacts / 'audit/resource_helpers', worktree)
        require(set(NEW) <= {r['mutator'] for r in rows.values()}, 'New mutant population incomplete')
        if variant == 'or':
            require(all(r['mutator'] in NEW for r in rows.values()), 'Unexpected operator in new-only report')
        else:
            manifest['compatibility'] = compatibility(rows, read(worktree / 'docs/development/mutation-ci-baselines/resource_helpers.json'))
        manifest['fingerprint'] = fingerprint(worktree, 'resource_helpers', audit['execution'])
        manifest['mutants'] = [classify(r) for _, r in sorted(rows.items())]
        manifest['per_operator'] = {op: dict(Counter(r['status'] for r in rows.values() if r['mutator'] == op))
                                    for op in sorted({r['mutator'] for r in rows.values()})}
        build = worktree / 'out/build/unixlike-clang-mut/mutation-builds/resource_helpers'
        manifest['binary_sha256'] = {p.relative_to(build).as_posix(): sha(p) for p in build.rglob('vkexec_core_tests') if p.is_file()}
        require(manifest['binary_sha256'], 'Missing instrumented executable')
        binary = build / 'test/vkexec_core_tests'
        diagnostics = artifacts / 'coverage'
        command([sys.executable, 'scripts/compare_mull_coverage.py', str(binary), str(worktree / CONFIG),
                 str(diagnostics), '--', '[resource-helpers]', '--order', 'lex', '--abort'],
                worktree, artifacts / 'coverage.log')
        command(['llvm-cov', 'export', str(binary),
                 '-instr-profile=' + str(artifacts / 'audit/resource_helpers/inclusive.profdata')],
                worktree, artifacts / 'coverage-export.json')
        diagnostic = diagnose(read(artifacts / 'audit/resource_helpers/inclusive.json'),
                              read(artifacts / 'coverage-export.json'), worktree,
                              (diagnostics / 'indexed.log').read_text())
        (artifacts / 'attribution.json').write_text(json.dumps(diagnostic, indent=2) + '\n')
        by_id = {normalize(r, worktree): r for r in diagnostic['locations']}
        for row in manifest['mutants']:
            detail = by_id[row['id']]
            names = {instance['function'] for instance in detail['instances']}
            row['function'] = next(iter(names)) if len(names) == 1 else None
            if names and all('12source_scope' in name for name in names):
                row['function'] = 'source_scope() accumulation lambda'
            row['evidence'] = ['audit/resource_helpers/inclusive.sqlite', 'attribution.json', 'coverage/indexed.log']
            if (row['status'] == 'not_covered' and detail['classification'] == 'source_covered_attribution_discrepancy'
                    and detail.get('mull_uncovered_ranges')
                    and any(i['count'] > 0 and i['containing_gap_regions'] for i in detail['instances'])):
                row['classification'] = 'known attribution limitation'
            if (row['status'] == 'survived' and row['id'] ==
                    'cxx_lt_to_le:include/vkexec/detail/resource_state_tracker_helpers.hpp:409:35:409:36'):
                row['classification'] = 'potential equivalence'
                row['evidence'].append('docs/development/mutation-resource-helpers-baseline.md')
                row['proposed_assertion'] = 'No distinguishing assertion identified: the extra terminal iteration has an empty inner loop (existing PR 8 analysis).'
        manifest['failed'] = not manifest.get('compatibility', {'compatible': True})['compatible']
    except Exception as error:
        manifest['error'] = str(error)
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - started
        try:
            if added:
                command(['git', 'worktree', 'remove', '--force', str(worktree)], root, artifacts / 'cleanup.log')
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except Exception as error:
            manifest['cleanup_error'] = str(error)
            manifest['failed'] = True
        (artifacts / 'experiment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def compare_experiments(manifests):
    require(len(manifests) >= 2, 'At least two experimental snapshots required')
    reference = manifests[0]
    populations = []
    configurations = {}
    for manifest in manifests:
        require(not manifest['failed'], 'Cannot compare failed experiment')
        for field in ('revision', 'toolchain_versions', 'test_args'):
            require(manifest[field] == reference[field], f'Experiment {field} mismatch')
        if 'fingerprint' in reference:
            for field in ('source_sha256', 'test_source_sha256', 'toolchain', 'build_options', 'timeout_policy', 'nix_lock_sha256'):
                require(manifest['fingerprint'][field] == reference['fingerprint'][field], f'Fingerprint {field} mismatch')
            variant = manifest['variant']
            digest = manifest['configuration_sha256']
            require(configurations.setdefault(variant, digest) == digest, 'Configuration changed between repeats')
        rows = {}
        for row in manifest['mutants']:
            require(row['classification'] in CLASSIFICATIONS, 'Unexpected outcome classification')
            require(row['file'] in sources('resource_helpers'), 'Unexpected source file')
            identifier = normalize(row, Path('/unused'))  # IDs must already be relative.
            require(identifier not in rows, 'Duplicate experiment ID')
            require(row['status'] in ('killed', 'survived', 'not_covered', 'timed_out', 'crashed', 'abnormal_exit'), 'Unknown status')
            rows[identifier] = row
        subset = {i: r['status'] for i, r in rows.items() if r['mutator'] in NEW}
        require(subset, 'Empty OR-operator population')
        populations.append(subset)
    return {'stable': all(p == populations[0] for p in populations),
            'wrapper_changed': any(m['wrapper_sha256'] != reference['wrapper_sha256'] for m in manifests),
            'runs': len(manifests), 'new_mutants': len(populations[0]),
            'differences': [sorted(i for i in populations[0].keys() | p.keys()
                                    if populations[0].get(i) != p.get(i)) for p in populations[1:]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['or', 'combined'])
    parser.add_argument('--compare', type=Path, nargs='+', help='Compare retained experiment.json snapshots by exact OR IDs and statuses')
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--artifacts', type=Path)
    args = parser.parse_args()
    try:
        if args.compare:
            result = compare_experiments([read(p) for p in args.compare])
            print(json.dumps(result, indent=2))
            return int(not result['stable'])
        require(args.variant and args.artifacts, '--variant and --artifacts are required for measurement')
        return int(experiment(Path.cwd(), args.variant, args.revision, args.artifacts)['failed'])
    except (ValueError, OSError, subprocess.CalledProcessError, sqlite3.Error) as error:
        print(f'Operator experiment failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
