from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import gpu_mutation_experiment as gpu

ROOT = Path(__file__).resolve().parents[3]


class Runner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'source'
        self.root.mkdir()
        for args in (('init', '-q'), ('config', 'user.email', 'test@example.com'),
                     ('config', 'user.name', 'Test')):
            subprocess.run(['git', '-C', str(self.root), *args], check=True)
        for name in ('.gitignore', 'CMakeLists.txt', 'test/CMakeLists.txt', gpu.CONFIG, gpu.SOURCE, 'flake.lock', 'cmake/mull/dag.yml', 'CMakePresets.json', 'cmake/MutationTesting.cmake', 'cmake/RunMutationScope.cmake', 'docs/development/mutation-ci-baselines/dag.json'):
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / name).read_bytes())
        subprocess.run(['git', '-C', str(self.root), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(self.root), 'commit', '-qm', 'fixture'], check=True)

    def test_clean_revision_and_dirty(self):
        self.assertEqual(len(gpu.clean_revision(self.root, 'HEAD')), 40)
        (self.root / 'unexpected').touch()
        with self.assertRaisesRegex(ValueError, 'clean committed'):
            gpu.clean_revision(self.root, 'HEAD')

    def test_exact_overlay(self):
        expected = gpu.overlay_texts(self.root)
        for path, content in expected.items():
            (self.root / path).write_text(content)
        gpu.check_overlay(self.root, expected)
        (self.root / 'test/CMakeLists.txt').write_text(expected['test/CMakeLists.txt'] + '\n')
        with self.assertRaisesRegex(ValueError, 'Overlay content'):
            gpu.check_overlay(self.root, expected)
        (self.root / 'CMakeLists.txt').write_text('unexpected')
        with self.assertRaisesRegex(ValueError, 'registration'):
            gpu.overlay_texts(self.root)

    def test_unexpected_worktree_files(self):
        expected = gpu.overlay_texts(self.root)
        for p, content in expected.items():
            (self.root / p).write_text(content)
        (self.root / gpu.SOURCE).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'tracked worktree'):
            gpu.check_overlay(self.root, expected)

    def execute_ready_shell(self, source, log):
        # Start the short execution timeout only after the fixture has flushed
        # its readiness line. Interpreter/process startup is not under test.
        real_popen = subprocess.Popen

        def launch(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            real_wait = process.wait
            ready = False

            def wait(timeout=None):
                nonlocal ready
                if not ready:
                    ready = True  # Cleanup must bypass the readiness wait on failure.
                    deadline = time.monotonic() + 10
                    while not log.read_text().endswith('\n'):
                        if time.monotonic() >= deadline:
                            self.fail('Shell fixture did not signal readiness within 10 seconds')
                        time.sleep(0.01)
                return real_wait(timeout=timeout)

            process.wait = wait
            return process

        with patch.object(gpu.subprocess, 'Popen', side_effect=launch):
            gpu.execute(['/bin/sh', '-c', source], self.root, log, timeout=0.2)

    def test_timeout_and_crash_retain_logs(self):
        for source, error in (("printf 'partial\\n'; exec sleep 60", subprocess.TimeoutExpired),
                              ("printf 'partial\\n'; kill -KILL \"$$\"", subprocess.CalledProcessError)):
            with self.subTest(error=error.__name__):
                log = Path(self.tmp.name) / 'audit.log'
                with self.assertRaises(error):
                    self.execute_ready_shell(source, log)
                self.assertEqual(log.read_text(), 'partial\n')

    @patch.dict(os.environ, {'MULL_PLUGIN': __file__})
    def test_cleanup_and_manifest_after_failure(self):
        artifacts = Path(self.tmp.name) / 'evidence'
        def command(args, cwd, log, **kwargs):
            if args[0] == 'git':
                return gpu.execute(args, cwd, log, **kwargs)
            log.write_text('infrastructure failed')
            raise subprocess.CalledProcessError(86, args)
        result = gpu.experiment(self.root, 'HEAD', artifacts, command=command, versions=lambda: {})
        self.assertEqual(result['state'], 'infrastructure_failure')
        self.assertTrue(result['worktree_removed'])
        self.assertTrue((artifacts / 'configure.log').exists())
        self.assertTrue((artifacts / 'experiment.json').exists())
        self.assertEqual(len(gpu.git(self.root, 'worktree', 'list').splitlines()), 1)
        self.assertEqual(gpu.git(self.root, 'status', '--porcelain'), '')

    @patch.dict(os.environ, {'MULL_PLUGIN': __file__})
    def test_audit_timeout_preserves_partial_reports_and_checks_health(self):
        import json
        from test_device_policy import DEVICE
        artifacts = Path(self.tmp.name) / 'timeout-evidence'
        calls = []
        def command(args, cwd, log, **kwargs):
            args = list(map(str, args))
            calls.append(args)
            if args[0] == 'git':
                return gpu.execute(args, cwd, log, **kwargs)
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text('partial output')
            base = Path(cwd) / 'out/build/unixlike-clang-mut'
            private = base / 'mutation-builds/gpu_graph_execution'
            binary = private / 'test/vkexec_vma_tests'
            private.mkdir(parents=True, exist_ok=True)
            (private / 'CMakeCache.txt').write_text(''.join(k + ':STRING=' + v + '\n' for k, v in gpu.OPTIONS.items()))
            if '--target' in args and 'vkexec_vma_tests' in args:
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_bytes(b'fixture binary')
            if args[0].endswith('vkexec_gpu_preflight'):
                Path(args[1]).write_text(json.dumps(DEVICE))
            if '--reporter' in args:
                Path(args[-1]).write_text('<testsuites><testcase/><testcase/></testsuites>')
            if 'vkexec_mutation_gpu_graph_execution' in args:
                reports = base / 'mutation-reports'
                reports.mkdir(parents=True)
                for suffix in ('sqlite', 'profraw', 'profdata'):
                    (reports / ('vkexec_core_gpu_graph_execution.' + suffix)).write_bytes(b'partial retained')
                raise subprocess.TimeoutExpired(args, 1)
            return 0.01
        result = gpu.experiment(self.root, 'HEAD', artifacts, command=command, versions=lambda: {})
        self.assertEqual(result['error_type'], 'TimeoutExpired')
        self.assertTrue(result['worktree_removed'])
        for filename in ('report.sqlite', 'profile.profraw', 'profile.profdata', 'post-test.xml', 'post-device.json'):
            self.assertTrue((artifacts / 'covered' / filename).is_file(), filename)
        self.assertFalse((artifacts / 'inclusive').exists())

    @patch.dict(os.environ, {'MULL_PLUGIN': __file__})
    def test_complete_orchestration(self):
        import json
        import sqlite3
        from contextlib import closing
        from test_device_policy import DEVICE
        artifacts = Path(self.tmp.name) / 'complete-evidence'
        audit_targets = []
        def command(args, cwd, log, **kwargs):
            args = list(map(str, args))
            if args[0] == 'git':
                return gpu.execute(args, cwd, log, **kwargs)
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text('successful fixture')
            base = Path(cwd) / 'out/build/unixlike-clang-mut'
            private = base / 'mutation-builds/gpu_graph_execution'
            private.mkdir(parents=True, exist_ok=True)
            (private / 'CMakeCache.txt').write_text(''.join(k + ':STRING=' + v + '\n' for k, v in gpu.OPTIONS.items()))
            if '--target' in args and 'vkexec_vma_tests' in args:
                binary = private / 'test/vkexec_vma_tests'
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_bytes(b'fixture binary')
            if args[0].endswith('vkexec_gpu_preflight'):
                Path(args[1]).write_text(json.dumps(DEVICE))
            if '--reporter' in args:
                Path(args[-1]).write_text('<testsuites><testcase/><testcase/></testsuites>')
            if '--target' in args and args[-1].startswith('vkexec_mutation_'):
                audit_targets.append(args[-1])
                reports = base / 'mutation-reports'
                reports.mkdir(parents=True, exist_ok=True)
                name = args[-1].replace('vkexec_mutation_', 'vkexec_core_')
                for suffix in ('profraw', 'profdata'):
                    (reports / (name + '.' + suffix)).write_bytes(b'profile')
                with closing(sqlite3.connect(reports / (name + '.sqlite'))) as db:
                    db.execute('CREATE TABLE mutant(filename, directory, execution_status, mutator, line_number, column_number, mutant_id)')
                    db.execute('INSERT INTO mutant VALUES (?, ?, 1, ?, 1, 2, ?)',
                               (gpu.SOURCE, str(cwd), 'cxx_lt_to_le', f'cxx_lt_to_le:{gpu.SOURCE}:1:2:1:3'))
                    db.commit()
            return 0.01
        result = gpu.experiment(self.root, 'HEAD', artifacts, command=command, versions=lambda: {})
        self.assertEqual(result['state'], 'measured_pending_review', result.get('error'))
        self.assertTrue(result['worktree_removed'])
        self.assertEqual(len(result['mutants']), 1)
        self.assertEqual(audit_targets, ['vkexec_mutation_gpu_graph_execution', 'vkexec_mutation_gpu_graph_execution_uncovered'])
        self.assertIsNone(result['mutation_score'])
        self.assertTrue((artifacts / 'comparison.json').exists())
        self.assertEqual(len(list((artifacts / 'baseline').glob('*.xml'))), 3)

    def test_timeout_kills_child_process(self):
        log = Path(self.tmp.name) / 'child.log'
        # Readiness deliberately takes longer than the execution timeout.
        source = "sleep 0.5; sleep 60 & printf '%s\\n' \"$!\"; wait"
        with self.assertRaises(subprocess.TimeoutExpired):
            self.execute_ready_shell(source, log)
        pid = int(log.read_text().strip())
        status = Path(f'/proc/{pid}/status')
        # SIGKILL delivery to the child can lag behind reaping the parent.
        deadline = time.monotonic() + 5
        while status.exists():
            try:
                state = status.read_text()
            except FileNotFoundError:
                break
            # Orphaned zombies can remain until the container init reaps them.
            if 'Z (zombie)' in state:
                break
            if time.monotonic() >= deadline:
                self.fail(f'Child process {pid} survived process-group termination: {state}')
            time.sleep(0.01)

    def test_actual_cache_options(self):
        p = Path(self.tmp.name) / 'CMakeCache.txt'
        p.write_text(''.join(k + ':STRING=' + v + '\n' for k, v in gpu.OPTIONS.items()) +
                     'CMAKE_CXX_FLAGS:STRING=-g\nvkexec_SOURCE_DIR:STATIC=/temporary/source\n')
        options = gpu.cache_options(p)
        self.assertEqual(options['CMAKE_CXX_FLAGS'], '-g')
        self.assertNotIn('vkexec_SOURCE_DIR', options)
        p.write_text(p.read_text().replace('vkexec_MULL_WORKERS:STRING=1', 'vkexec_MULL_WORKERS:STRING=2'))
        with self.assertRaisesRegex(ValueError, 'Actual CMake options'):
            gpu.cache_options(p)

    def test_untracked_and_staged_overlay_changes(self):
        expected = gpu.overlay_texts(self.root)
        for p, content in expected.items():
            (self.root / p).write_text(content)
        p = self.root / 'unexpected'
        p.touch()
        with self.assertRaisesRegex(ValueError, 'untracked'):
            gpu.check_overlay(self.root, expected)
        p.unlink()
        subprocess.run(['git', '-C', str(self.root), 'add', 'CMakeLists.txt'], check=True)
        with self.assertRaisesRegex(ValueError, 'staged'):
            gpu.check_overlay(self.root, expected)
