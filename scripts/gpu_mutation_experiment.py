"""Opt-in isolated GPU mutation measurement. Never updates canonical baselines."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import signal
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_mutation_baselines import require, read, sha, rows_map
from mutation_operator_experiment import git, tool_versions
from validate_mull_report import parse_report
from diagnose_mull_coverage import diagnose

SOURCE = 'include/vkexec/detail/pass_execution.hpp'
CONFIG = 'cmake/mull/experiments/gpu_graph_execution.yml'
ARGS = ['[mutation-gpu]', '--order', 'lex', '--abort']
MUTATORS = {'cxx_eq_to_ne', 'cxx_ne_to_eq', 'cxx_ge_to_gt', 'cxx_ge_to_lt',
            'cxx_gt_to_ge', 'cxx_gt_to_le', 'cxx_le_to_gt', 'cxx_le_to_lt',
            'cxx_lt_to_ge', 'cxx_lt_to_le'}
OPTIONS = {'vkexec_BUILD_VMA': 'ON', 'vkexec_BUILD_TOOLS': 'OFF',
           'vkexec_MULL_WORKERS': '1', 'vkexec_MULL_TIMEOUT_MS': '60000',
           'vkexec_MULL_MINIMUM_TIMEOUT_MS': '5000'}
TEST_OVERLAY = '''
# PR 12 temporary GPU experiment overlay.
target_sources(vkexec_vma_tests PRIVATE "${PROJECT_SOURCE_DIR}/test/mutation/gpu/graph_execution_tests.cpp")
add_executable(vkexec_gpu_preflight "${PROJECT_SOURCE_DIR}/test/mutation/gpu/gpu_preflight.cpp")
set_target_properties(vkexec_gpu_preflight PROPERTIES CXX_SCAN_FOR_MODULES OFF)
target_link_libraries(vkexec_gpu_preflight PRIVATE vkexec::vma)
'''
SCOPE_OVERLAY = '''vkexec_register_mutation_scope(
        NAME gpu_graph_execution
        CONFIG "${PROJECT_SOURCE_DIR}/cmake/mull/experiments/gpu_graph_execution.yml"
        TEST_TARGET vkexec_vma_tests
        SOURCES "include/vkexec/detail/pass_execution.hpp"
        TEST_ARGS "[mutation-gpu]" "--order" "lex" "--abort")
      '''


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def execute(command, cwd, log, env=None, timeout=1800):
    """Retain output and kill descendants even when an audit's parent has exited."""
    log.parent.mkdir(parents=True, exist_ok=True)
    log.with_suffix(log.suffix + '.command.json').write_text(json.dumps(list(map(str, command))))
    started = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stream.write('GPU_INFRASTRUCTURE_FAILURE: process group did not terminate after SIGKILL\n')
                stream.flush()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)
    return time.monotonic() - started


def clean_revision(root, revision):
    require(not git(root, 'status', '--porcelain', '--untracked-files=all'),
            'Measurement requires a clean committed source checkout')
    return git(root, 'rev-parse', '--verify', '--end-of-options', revision + '^{commit}')


def overlay_texts(root):
    top = (root / 'CMakeLists.txt').read_text()
    tests = (root / 'test/CMakeLists.txt').read_text()
    anchor = 'vkexec_finalize_mutation_scopes()'
    require(top.count(anchor) == 1 and 'gpu_graph_execution' not in top, 'Unexpected mutation registration')
    require(tests.count('    integration/resource_region_tests.cpp)') == 1
            and 'vkexec_gpu_preflight' not in tests, 'Unexpected VMA test registration')
    # Append after the existing tests, where vkexec_vma_tests is already defined.
    return {'CMakeLists.txt': top.replace(anchor, SCOPE_OVERLAY + anchor),
            'test/CMakeLists.txt': tests + TEST_OVERLAY}


def check_overlay(root, expected):
    require(set(git(root, 'diff', '--name-only', 'HEAD').splitlines()) == set(expected),
            'Unexpected tracked worktree changes')
    require(not git(root, 'ls-files', '--others', '--exclude-standard'), 'Unexpected untracked worktree changes')
    require(not git(root, 'diff', '--cached', '--name-only'), 'Unexpected staged worktree changes')
    for path, content in expected.items():
        require((root / path).read_text() == content, 'Overlay content changed: ' + path)


def device_policy(device, allow_software=False):
    required = {'device_name', 'vendor_id', 'device_id', 'device_type', 'api_version',
                'driver_version', 'queue_family', 'software_device', 'identity'}
    require(required <= device.keys(), 'Missing device properties')
    require(type(device['software_device']) is bool and type(device['queue_family']) is int
            and device['queue_family'] >= 0 and bool(device['identity']), 'Invalid device properties')
    require(allow_software or not device['software_device'], 'Only software Vulkan: hardware result inconclusive')
    return device


def validate_baseline(path):
    root = ET.parse(path).getroot()
    for suite in root.iter('testsuite'):
        require(all(int(suite.get(field, '0')) == 0 for field in ('skipped', 'failures', 'errors')),
                'GPU baseline failed or skipped')
    cases = list(root.iter('testcase'))
    require(len(cases) == 2, 'Expected exactly two GPU tests; empty selection or aborted baseline')
    require(all(c.get('status') not in ('skipped', 'notrun') and not list(c.iter('skipped')) and not list(c.iter('failure')) and not list(c.iter('error'))
                for c in cases), 'GPU baseline failed or skipped')


def empty_covered_validator_failure(error, directory):
    # The canonical validator requires a nonempty covered population. An empty
    # covered SQLite can still be checked against the nonempty inclusive audit.
    if not isinstance(error, subprocess.CalledProcessError):
        return False
    if 'Mull report validation failed: Mull report contains no mutants' not in (directory / 'mull.log').read_text():
        return False
    database = directory / 'report.sqlite'
    if not database.is_file() or (directory / 'report.json').exists():
        return False
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        return db.execute('SELECT filename, directory, execution_status, mutator, line_number, column_number, mutant_id FROM mutant LIMIT 1').fetchone() is None


def load_reports(artifacts, root):
    reports = []
    for mode in ('covered', 'inclusive'):
        directory = artifacts / mode
        for suffix in ('profraw', 'profdata'):
            profile = directory / ('profile.' + suffix)
            require(profile.is_file() and profile.stat().st_size > 0, 'Missing profile: ' + str(profile))
        database = directory / 'report.sqlite'
        # Empty covered populations are valid if inclusive contains only uncovered IDs.
        require(database.is_file(), 'Missing SQLite report')
        with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as db:
            empty = db.execute('SELECT filename, directory, execution_status, mutator, line_number, column_number, mutant_id FROM mutant LIMIT 1').fetchone() is None
        require(not empty or mode == 'covered', 'Empty inclusive mutant population')
        raw = {'locations': []} if empty else parse_report(database, root / SOURCE, root)
        generated = directory / 'report.json'
        if generated.exists():
            require(read(generated) == raw, 'SQLite/JSON contradiction')
        dump(generated, raw)
        rows = rows_map(raw['locations'], 'dag', root) if raw['locations'] else {}
        require(all(r['mutator'] in MUTATORS for r in rows.values()), 'Unexpected mutation operator')
        reports.append(rows)
    covered, inclusive = reports
    require(covered == {i: r for i, r in inclusive.items() if r['status'] != 'not_covered'},
            'Covered/inclusive exact ID or status mismatch')
    # Mull's killed enum does not prove an assertion fired: retain as unreviewed.
    for row in inclusive.values():
        row['detection_review'] = ('unverified: assertion versus infrastructure requires per-mutant replay'
                                   if row['status'] == 'killed' else 'not an assertion detection')
    return inclusive


def compare_cpu(gpu, cpu):
    cpu = rows_map(cpu['mutants'], 'dag', Path('/unused'))
    result = []
    for identifier in sorted(gpu.keys() | cpu.keys()):
        g, c = gpu.get(identifier), cpu.get(identifier)
        gs, cs = g['status'] if g else None, c['status'] if c else None
        if g is None:
            label = 'CPU-only mutant'
        elif c is None:
            label = 'GPU-only mutant'
        elif gs == 'not_covered':
            label = 'Uncovered by GPU'
        elif gs == 'killed' and cs == 'killed':
            label = 'Killed by both (raw Mull status)'
        elif gs == 'killed' and cs in ('survived', 'not_covered'):
            label = f'{cs} CPU, killed GPU (requires assertion attribution)'
        elif cs == 'killed' and gs == 'survived':
            label = 'Killed CPU, survived GPU'
        else:
            label = 'Other intersecting outcome'
        result.append({'id': identifier, 'cpu': cs, 'gpu': gs, 'classification': label})
    return {'intersection': len(gpu.keys() & cpu.keys()), 'outcomes': result,
            'confirmed_incremental_detections': None, 'mutation_score': None}


def compare_runs(manifests):
    require(len(manifests) == 3, 'Three independent experiments required')
    first = manifests[0]
    for m in manifests:
        require(m['state'] == 'measured_pending_review', 'Cannot compare failed/incomplete measurements')
        for key in ('revision', 'versions', 'device', 'fingerprint', 'hardware_measurement', 'environment'):
            require(m[key] == first[key], 'Incomparable runs: ' + key)
    populations = [{i: r['status'] for i, r in rows_map(m['mutants'], 'dag', Path('/unused')).items()}
                   for m in manifests]
    ids = set().union(*populations)
    unstable = [i for i in sorted(ids) if len({p.get(i) for p in populations}) != 1]
    return {'stable': not unstable, 'unstable_ids': unstable, 'runs': 3, 'mutation_score': None}


def cache_options(path):
    options = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(('#', '//')) or '=' not in line or ':' not in line.split('=', 1)[0]:
            continue
        key = line.split(':', 1)[0]
        if ((key.startswith('vkexec_') and not key.endswith(('_DIR', '_SCOPE')))
                or key in ('BUILD_TESTING', 'BUILD_SHARED_LIBS', 'CMAKE_BUILD_TYPE', 'CMAKE_CXX_COMPILER')
                or key.startswith(('CMAKE_CXX_FLAGS', 'CMAKE_EXE_LINKER_FLAGS', 'CMAKE_SHARED_LINKER_FLAGS'))):
            options[key] = line.split('=', 1)[1]
    require(all(options.get(k) == v for k, v in OPTIONS.items()), 'Actual CMake options differ from requested experiment')
    return options


def experiment(root, revision, artifacts, allow_software=False, outer_timeout=1800,
               command=execute, versions=tool_versions):
    root, artifacts = root.resolve(), artifacts.resolve()
    commit = clean_revision(root, revision)
    require(not artifacts.exists(), 'Evidence destination already exists')
    artifacts.mkdir(parents=True)
    manifest = {'schema_version': 1, 'kind': 'noncanonical GPU experiment', 'revision': commit,
                'scope': 'gpu_graph_execution', 'test_target': 'vkexec_vma_tests',
                'source_allowlist': [SOURCE], 'operators': ['cxx_comparison', 'cxx_boundary'],
                'source_clean_before_overlay': True,
                'state': 'infrastructure_failure', 'mutants': [], 'mutation_score': None,
                'hardware_measurement': not allow_software, 'timings': {}, 'audit_execution': {},
                'environment': {k: v for k, v in os.environ.items() if k.startswith(('VK_', 'MESA_', '__NV_')) or k in
                    ('CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'NIX_CFLAGS_COMPILE', 'NIX_LDFLAGS')}}
    parent = None
    worktree = None
    added = False
    build = None
    started = time.monotonic()
    env = dict(os.environ)
    # Ignore inherited experimental identity/software overrides unless explicitly requested.
    env.pop('MULL_CONFIG', None)
    auxiliary_profile = str(artifacts / 'auxiliary-%p.profraw')
    env['LLVM_PROFILE_FILE'] = auxiliary_profile
    env.pop('VKEXEC_GPU_IDENTITY', None)
    env.pop('VKEXEC_GPU_ALLOW_SOFTWARE', None)
    if allow_software:
        env['VKEXEC_GPU_ALLOW_SOFTWARE'] = '1'
    def run(args, log, timeout=outer_timeout):
        step_started = time.monotonic()
        try:
            command(args, worktree, artifacts / log, env=env, timeout=timeout)
        finally:
            manifest['timings'][log] = time.monotonic() - step_started
    def preflight(log):
        try:
            run([build / 'test/vkexec_gpu_preflight', artifacts / log], log + '.log')
        except subprocess.SubprocessError:
            if (artifacts / log).is_file() and (artifacts / log).stat().st_size:
                manifest['device'] = read(artifacts / log)
                dump(artifacts / 'device.json', manifest['device'])
            if 'software device rejected' in (artifacts / (log + '.log')).read_text():
                manifest['state'] = 'hardware_inconclusive'
            raise
        device = device_policy(read(artifacts / log), allow_software)
        if 'device' in manifest:
            require(device == manifest['device'], 'Device changed during experiment')
        else:
            manifest['device'] = device
            dump(artifacts / 'device.json', device)
            env['VKEXEC_GPU_IDENTITY'] = device['identity']
    try:
        manifest['versions'] = versions()
        dump(artifacts / 'versions.json', manifest['versions'])
        parent = Path(tempfile.mkdtemp(prefix='vkexec-gpu-'))
        worktree = parent / 'source'
        command(['git', 'worktree', 'add', '--detach', worktree, commit], root,
                artifacts / 'worktree.log', timeout=outer_timeout)
        added = True
        expected = overlay_texts(worktree)
        for path, content in expected.items():
            (worktree / path).write_text(content)
        check_overlay(worktree, expected)
        (artifacts / 'cmake-overlay.patch').write_text(git(worktree, 'diff', '--binary') + '\n')
        manifest['fingerprint'] = {
            'source_sha256': sha(worktree / SOURCE), 'configuration_sha256': sha(worktree / CONFIG),
            'fixture_sha256': {p.name: sha(p) for p in sorted((worktree / 'test/mutation/gpu').glob('*'))},
            'overlay_sha256': {p: sha(worktree / p) for p in expected},
            'wrapper_sha256': sha(Path(__file__)), 'nix_lock_sha256': sha(worktree / 'flake.lock'),
            'build_configuration_sha256': {p: sha(worktree / p) for p in
                ('CMakePresets.json', 'cmake/MutationTesting.cmake', 'cmake/RunMutationScope.cmake')},
            'mull_plugin_sha256': sha(Path(os.environ['MULL_PLUGIN'])),
            'options': OPTIONS, 'test_args': ARGS, 'outer_timeout_seconds': outer_timeout}
        require((worktree / CONFIG).read_text() == (worktree / 'cmake/mull/dag.yml').read_text(),
                'GPU configuration must match DAG operators and allowlist exactly')
        run(['cmake', '--preset', 'unixlike-clang-mut', *['-D' + k + '=' + v for k, v in OPTIONS.items()]], 'configure.log')
        base = worktree / 'out/build/unixlike-clang-mut'
        build = base / 'mutation-builds/gpu_graph_execution'
        run(['cmake', '-S', worktree, '-B', build, '-G', 'Ninja', '-C', base / 'mutation-configs/cache.cmake',
             '-Dvkexec_MULL_INTERNAL_SCOPE:INTERNAL=gpu_graph_execution',
             '-Dvkexec_MULL_REPORT_DIR:INTERNAL=' + str(base / 'mutation-reports')], 'private-configure.log')
        manifest['fingerprint']['actual_build_options'] = cache_options(build / 'CMakeCache.txt')
        run(['cmake', '--build', build, '--parallel', '2', '--target', 'vkexec_vma_tests', 'vkexec_gpu_preflight'], 'build.log')
        binary = build / 'test/vkexec_vma_tests'
        manifest['binary_sha256'] = sha(binary)
        preflight('preflight.json')
        for index in range(1, 4):
            xml = artifacts / f'baseline/run-{index}.xml'
            env['LLVM_PROFILE_FILE'] = str(artifacts / f'baseline/run-{index}.profraw')
            run([binary, *ARGS, '--reporter', 'junit', '--out', xml], f'baseline/run-{index}.log')
            validate_baseline(xml)
        env['LLVM_PROFILE_FILE'] = auxiliary_profile
        for mode, suffix in (('covered', ''), ('inclusive', '_uncovered')):
            directory = artifacts / mode
            directory.mkdir()
            name = 'vkexec_core_gpu_graph_execution' + suffix
            audit_error = None
            try:
                run(['cmake', '--build', base, '--target', 'vkexec_mutation_gpu_graph_execution' + suffix], mode + '/mull.log')
                manifest['audit_execution'][mode] = {'returncode': 0, 'timed_out': False}
            except (subprocess.SubprocessError, OSError) as error:
                audit_error = error
                manifest['audit_execution'][mode] = {
                    'returncode': getattr(error, 'returncode', None),
                    'timed_out': isinstance(error, subprocess.TimeoutExpired),
                    'error': str(error)}
            finally:
                # Preserve partial reports even after a process-group timeout or crash.
                for ext, dest in (('sqlite', 'report.sqlite'), ('json', 'report.json'),
                                  ('profraw', 'profile.profraw'), ('profdata', 'profile.profdata')):
                    src = base / 'mutation-reports' / (name + '.' + ext)
                    if src.is_file():
                        shutil.copy2(src, directory / dest)
            preflight(mode + '/post-device.json')
            xml = directory / 'post-test.xml'
            env['LLVM_PROFILE_FILE'] = str(directory / 'post-test.profraw')
            run([binary, *ARGS, '--reporter', 'junit', '--out', xml], mode + '/post-test.log')
            validate_baseline(xml)
            env['LLVM_PROFILE_FILE'] = auxiliary_profile
            if audit_error is not None:
                if mode == 'covered' and empty_covered_validator_failure(audit_error, directory):
                    manifest['covered_validator_limitation'] = 'Empty covered SQLite independently checked against inclusive IDs'
                else:
                    raise audit_error
        rows = load_reports(artifacts, worktree)
        manifest['mutants'] = [r for _, r in sorted(rows.items())]
        dump(artifacts / 'comparison.json', compare_cpu(rows, read(worktree / 'docs/development/mutation-ci-baselines/dag.json')))
        if any(r['status'] == 'not_covered' for r in rows.values()):
            run(['llvm-cov', 'export', binary,
                 '-instr-profile=' + str(artifacts / 'inclusive/profile.profdata')], 'coverage-export.json')
            diagnostic = diagnose(read(artifacts / 'inclusive/report.json'),
                                  read(artifacts / 'coverage-export.json'), worktree,
                                  (artifacts / 'inclusive/mull.log').read_text())
            dump(artifacts / 'attribution.json', diagnostic)
        check_overlay(worktree, expected)
        manifest['state'] = 'measured_pending_review'
        if any('GPU_INFRASTRUCTURE_FAILURE:' in (artifacts / mode / 'mull.log').read_text()
               for mode in ('covered', 'inclusive')):
            manifest['state'] = 'infrastructure_failure'
            manifest['error'] = 'Infrastructure failure appeared during mutant execution; raw kills are unverified'
    except Exception as error:
        manifest['error'] = str(error)
        manifest['error_type'] = type(error).__name__
    finally:
        if added:
            try:
                # Preserve compiler cache, partial profiles and build diagnostics before removing our tree.
                for filename in ('CMakeCache.txt', 'compile_commands.json'):
                    if build and (build / filename).is_file():
                        shutil.copy2(build / filename, artifacts / filename)
                command(['git', 'worktree', 'remove', '--force', worktree], root,
                        artifacts / 'cleanup.log', timeout=60)
                manifest['worktree_removed'] = True
            except Exception as error:
                manifest['cleanup_error'] = str(error)
                manifest['state'] = 'infrastructure_failure'
        if parent and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
        manifest['elapsed_seconds'] = time.monotonic() - started
        dump(artifacts / 'experiment.json', manifest)
        (artifacts / 'summary.md').write_text(
            '# GPU mutation experiment\n\nState: ' + manifest['state'] + '\n\n' +
            manifest.get('error', 'Raw Mull kills require assertion/driver attribution before claiming incremental value.') +
            '\n\nNo mutation score or permanent GPU CI recommendation is inferred.\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--allow-software', action='store_true', help='Diagnostic only; never a hardware result')
    parser.add_argument('--outer-timeout', type=int, default=1800)
    parser.add_argument('--compare', type=Path, nargs=3)
    args = parser.parse_args()
    try:
        if args.compare:
            result = compare_runs([read(p) for p in args.compare])
            print(json.dumps(result, indent=2))
            return int(not result['stable'])
        require(args.artifacts is not None and args.outer_timeout > 0, 'Fresh --artifacts and positive timeout required')
        result = experiment(Path.cwd(), args.revision, args.artifacts, args.allow_software, args.outer_timeout)
        print(json.dumps({'state': result['state'], 'error': result.get('error')}, indent=2))
        return int(result['state'] != 'measured_pending_review')
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
