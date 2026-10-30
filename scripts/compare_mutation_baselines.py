"""Strict per-mutant CI comparison. Exit 1: invalid; 2: regression; 3: review."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

STATUSES = ('killed', 'survived', 'not_covered', 'timed_out', 'crashed', 'abnormal_exit')
SCOPES = {
    'ranges': ('mull.yml', ['resource_ranges.hpp'], '[ranges]'),
    'dag': ('cmake/mull/dag.yml', ['pass_execution.hpp'], '[dag-plan]'),
    'resource_state': ('cmake/mull/resource_state.yml', ['resource_state_tracker.hpp'], '[resource-state]'),
    'resource_helpers': ('cmake/mull/resource_helpers.yml', ['resource_state_tracker_helpers.hpp', 'resource_state_tracker_validation.hpp'], '[resource-helpers]'),
    'barrier_coalescing': ('cmake/mull/barrier_coalescing.yml', ['resource_barrier_coalescing.hpp'], '[barrier-coalescing]'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    with Path(path).open() as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f'Expected JSON object: {path}')
    return value


def sources(scope):
    require(scope in SCOPES, f'Unknown scope: {scope}')
    return ['include/vkexec/detail/' + name for name in SCOPES[scope][1]]


def normalize(row, root):
    file = row['file']
    mutator = row['mutator']
    identifier = row.get('mutant_id', row.get('id'))
    require(isinstance(mutator, str) and mutator, 'Missing mutator')
    require(isinstance(identifier, str), 'Missing mutant ID')
    # Only remove this checkout's exact prefix, with an independently validated file.
    prefix = f'{mutator}:{Path(root).resolve()}/{file}:'
    if identifier.startswith(prefix):
        identifier = f'{mutator}:{file}:' + identifier[len(prefix):]
    require(identifier.startswith(f'{mutator}:{file}:'), f'Unexpected mutant ID path: {identifier}')
    tail = identifier[len(f'{mutator}:{file}:'):].split(':')
    require(len(tail) >= 4 and all(x.isdigit() and int(x) > 0 for x in tail[:4]), 'Invalid ID coordinates')
    require(type(row.get('line')) is int and type(row.get('column')) is int, 'Missing location')
    require([int(x) for x in tail[:2]] == [row['line'], row['column']], 'ID/location contradiction')
    return identifier


def rows_map(rows, scope, root):
    require(isinstance(rows, list) and rows, 'Empty or invalid mutant population')
    result = {}
    for row in rows:
        require(isinstance(row, dict), 'Invalid mutant record')
        require(row.get('file') in sources(scope), 'Unexpected source file')
        require(row.get('status') in STATUSES, 'Unknown status')
        identifier = normalize(row, root)
        require(identifier not in result, f'Duplicate normalized ID: {identifier}')
        result[identifier] = {**row, 'id': identifier}
        result[identifier].pop('mutant_id', None)
    require({r['file'] for r in result.values()} == set(sources(scope)), 'Selected source missing')
    return result


def check_counts(report, rows):
    require(isinstance(report, dict), 'Invalid summary object')
    counts = Counter(r['status'] for r in rows)
    require(type(report.get('total')) is int and report['total'] == len(rows), 'Total mismatch')
    for status in STATUSES:
        require(type(report.get(status)) is int and report[status] == counts[status], f'{status} count mismatch')


def report_map(report, scope, root):
    require(type(report.get('schema_version')) is int and report['schema_version'] == 2, 'Unsupported report schema')
    result = rows_map(report.get('locations'), scope, root)
    require(all('mutant_id' in row and 'id' not in row for row in report['locations']), 'Schema 2 requires mutant_id fields')
    check_counts(report, list(result.values()))
    for field, key in [('per_file', 'file'), ('per_mutator', 'mutator')]:
        require(isinstance(report.get(field), dict), f'Missing {field}')
        groups = {r[key] for r in result.values()}
        require(set(report[field]) == groups, f'{field} groups mismatch')
        for group in groups:
            check_counts(report[field][group], [r for r in result.values() if r[key] == group])
    return result


def validate_pair(covered, inclusive, scope, root):
    first = report_map(covered, scope, root)
    second = report_map(inclusive, scope, root)
    require(all(r['status'] != 'not_covered' for r in first.values()), 'Covered report contains uncovered mutant')
    expected = {i: r for i, r in second.items() if r['status'] != 'not_covered'}
    require(first == expected, 'Covered/inclusive population or status contradiction')
    require(any(r['status'] in ('killed', 'survived') for r in first.values()), 'No meaningfully evaluated mutants')
    return second


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(root, scope, execution):
    root = Path(root)
    config = SCOPES[scope][0]
    text = (root / config).read_text()
    operators = re.search(r'^mutators:\s*\n((?:\s+-[^\n]+\n)+)', text, re.M)
    require(operators is not None, 'Malformed mutation operator configuration')
    require(set(execution) == {'toolchain', 'build_options', 'timeout_policy'}, 'Malformed execution metadata')
    require(execution['toolchain'] == {'clang': '22.1.8', 'llvm': '22.1.8', 'mull': '0.35.0'}, 'Unexpected toolchain')
    return {
        'source_sha256': {p: sha(root / p) for p in sources(scope)},
        'config_sha256': sha(root / config),
        'operators': re.findall(r'^\s+-\s+(\S+)', operators[1], re.M),
        'toolchain': execution['toolchain'],
        'nix_lock_sha256': sha(root / 'flake.lock'),
        'test_args': [SCOPES[scope][2], '--order', 'lex', '--abort'],
        'test_source_sha256': {p.relative_to(root).as_posix(): sha(p) for p in sorted(p for p in (root / 'test').rglob('*') if p.suffix in ('.cpp', '.hpp', '.h'))},
        'test_registration_sha256': {'test/CMakeLists.txt': sha(root / 'test/CMakeLists.txt')},
        'build_options': execution['build_options'],
        'timeout_policy': execution['timeout_policy'],
        'infrastructure_sha256': {p: sha(root / p) for p in ['CMakeLists.txt', 'Dependencies.cmake', 'ProjectOptions.cmake', 'CMakePresets.json', 'cmake/MutationTesting.cmake', 'cmake/RunMutationScope.cmake', 'flake.nix']},
    }


def validate_fingerprint(fp, scope):
    required = {'source_sha256', 'config_sha256', 'operators', 'toolchain', 'nix_lock_sha256', 'test_args', 'test_source_sha256', 'build_options', 'timeout_policy', 'infrastructure_sha256'}
    require(isinstance(fp, dict) and set(fp) in (required, required | {'test_registration_sha256'}), 'Invalid fingerprint fields')
    require(set(fp['source_sha256']) == set(sources(scope)), 'Fingerprint source allowlist mismatch')
    if 'test_registration_sha256' in fp:
        require(set(fp['test_registration_sha256']) == {'test/CMakeLists.txt'}, 'Invalid test registration fingerprint')
        require('test/CMakeLists.txt' not in fp['infrastructure_sha256'], 'Duplicate test registration fingerprint')
    for field in ('source_sha256', 'test_source_sha256', 'infrastructure_sha256', *(['test_registration_sha256'] if 'test_registration_sha256' in fp else [])):
        require(isinstance(fp[field], dict) and fp[field], f'Empty {field}')
        for path, digest in fp[field].items():
            require(not Path(path).is_absolute() and '..' not in Path(path).parts, 'Unsafe fingerprint path')
            require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'Invalid hash')
    for field in ('config_sha256', 'nix_lock_sha256'):
        require(isinstance(fp[field], str) and re.fullmatch('[0-9a-f]{64}', fp[field]), 'Invalid hash')
    require(isinstance(fp['operators'], list) and fp['operators'] and all(isinstance(x, str) and x for x in fp['operators']), 'Invalid operators')
    require(fp['test_args'] == [SCOPES[scope][2], '--order', 'lex', '--abort'], 'Invalid test selection')
    require(isinstance(fp['toolchain'], dict) and set(fp['toolchain']) == {'clang', 'llvm', 'mull'} and all(isinstance(x, str) and x for x in fp['toolchain'].values()), 'Invalid toolchain')
    require(isinstance(fp['build_options'], dict) and fp['build_options'], 'Missing build options')
    require(isinstance(fp['timeout_policy'], dict) and set(fp['timeout_policy']) == {'warmup_ms', 'minimum_ms', 'workers'} and all(type(x) is int and x > 0 for x in fp['timeout_policy'].values()), 'Invalid timeout policy')


def split_test_registration(fp):
    """Read legacy snapshots without rewriting their measured provenance."""
    execution = {k: v for k, v in fp.items() if k not in ('test_source_sha256', 'test_registration_sha256')}
    execution['infrastructure_sha256'] = dict(fp['infrastructure_sha256'])
    legacy = execution['infrastructure_sha256'].pop('test/CMakeLists.txt', None)
    registration = fp.get('test_registration_sha256', {'test/CMakeLists.txt': legacy})
    return execution, registration


def compare(scope, baseline, covered, inclusive, root, current_fp, policy='regression'):
    require(policy in ('report', 'regression'), 'Invalid policy')
    require(type(baseline.get('schema_version')) is int and baseline['schema_version'] == 1, 'Unsupported baseline schema')
    require(baseline.get('scope') == scope, 'Baseline scope mismatch')
    require(isinstance(baseline.get('baseline_revision'), str) and re.fullmatch('[0-9a-f]{40}', baseline['baseline_revision']), 'Invalid baseline revision')
    validate_fingerprint(baseline.get('fingerprint'), scope)
    validate_fingerprint(current_fp, scope)
    before = rows_map(baseline.get('mutants'), scope, root)
    require(all(row.get('id') in before and row['id'] == normalize(row, root) and 'mutant_id' not in row for row in baseline['mutants']), 'Baseline IDs must be normalized id fields')
    after = validate_pair(covered, inclusive, scope, root)
    current_execution, current_registration = split_test_registration(current_fp)
    baseline_execution, baseline_registration = split_test_registration(baseline['fingerprint'])
    differences = [k for k in current_execution if current_execution[k] != baseline_execution[k]]
    changes = []
    for identifier in sorted(before.keys() | after.keys()):
        old = before.get(identifier, {}).get('status')
        new = after.get(identifier, {}).get('status')
        kind = 'unchanged'
        if old is None:
            kind = 'new'
        elif new is None:
            kind = 'removed'
        elif (new in ('timed_out', 'crashed', 'abnormal_exit') and new != old) or (old == 'killed' and new != 'killed') or (old == 'survived' and new == 'not_covered'):
            kind = 'regression'
        elif old != new:
            kind = 'improvement' if new == 'killed' else 'changed'
        changes.append({'id': identifier, 'file': (after.get(identifier) or before[identifier])['file'], 'previous': old, 'current': new, 'classification': kind})
    population = any(r['classification'] in ('new', 'removed') for r in changes)
    comparable = not differences and not population
    regressions = sum(r['classification'] == 'regression' for r in changes)
    status = 'review' if not comparable else 'failed' if regressions else 'passed'
    return {'schema_version': 1, 'scope': scope, 'policy': policy, 'status': status,
            'comparable': comparable, 'fingerprint_mismatches': differences,
            'tests_changed': current_fp['test_source_sha256'] != baseline['fingerprint']['test_source_sha256'],
            'test_registration_changed': current_registration != baseline_registration,
            'population_changed': population, 'regressions': regressions,
            'counts': {s: sum(r['status'] == s for r in after.values()) for s in STATUSES}, 'changes': changes,
            'exit_code': (3 if not comparable else 2 if regressions else 0) if policy == 'regression' else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=SCOPES, required=True)
    for name in ('baseline', 'covered', 'inclusive', 'repository-root', 'execution-metadata', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--policy', choices=['report', 'regression'], default='report')
    args = parser.parse_args()
    try:
        fp = fingerprint(args.repository_root, args.scope, read(args.execution_metadata))
        result = compare(args.scope, read(args.baseline), read(args.covered), read(args.inclusive), args.repository_root, fp, args.policy)
    except (ValueError, OSError, TypeError, KeyError) as error:
        result = {'scope': args.scope, 'status': 'invalid', 'error': str(error), 'exit_code': 1}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return result['exit_code']


if __name__ == '__main__':
    sys.exit(main())
