"""Exercise orchestration failure paths without requiring external tools."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
from test_comparison import report

spec = importlib.util.spec_from_file_location('audit_runner', ROOT / 'scripts/ci/run_mutation_audits.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_failed_target_still_collects_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Copy just the fingerprint inputs; no build or Vulkan dependencies.
            files = ['CMakeLists.txt', 'Dependencies.cmake', 'ProjectOptions.cmake', 'test/CMakeLists.txt', 'CMakePresets.json', 'cmake/MutationTesting.cmake', 'cmake/RunMutationScope.cmake', 'flake.nix', 'flake.lock']
            files += [s[0] for s in runner.SCOPES.values()]
            from compare_mutation_baselines import sources
            files += [s for scope in runner.SCOPES for s in sources(scope)]
            files += ['test/core/dag_planning_tests.cpp']
            for file in files:
                (root / file).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / file, root / file)
            build = root / 'out/build/unixlike-clang-mut'
            reports = build / 'mutation-reports'
            reports.mkdir(parents=True)
            calls = []

            def fake_run(command, log):
                log.write_text('diagnostic log\n')
                if command[:2] == ['cmake', '--preset']:
                    (build / 'CMakeCache.txt').write_text('CMAKE_BUILD_TYPE:STRING=Debug\n')
                if '--target' not in command:
                    return 0
                target = command[-1].removeprefix('vkexec_mutation_')
                calls.append(target)
                scope = target.removesuffix('_uncovered')
                rows = [{'file': source, 'line': 1, 'column': 1, 'mutator': 'cxx_eq_to_ne', 'mutant_id': f'cxx_eq_to_ne:{root}/{source}:1:1:1:2', 'status': 'killed'} for source in sources(scope)]
                stem = reports / ('vkexec_core_' + target)
                for extension in ('sqlite', 'txt', 'profraw', 'profdata'):
                    stem.with_suffix('.' + extension).write_text('evidence')
                stem.with_suffix('.json').write_text(json.dumps(report(rows)))
                return 1 if target == 'ranges' else 0

            original_is_file = Path.is_file
            env = {'MULL_PLUGIN': '/nix/store/fake/plugin', 'MULL_RUNNER': '/nix/store/fake/mull-runner-22', 'MULL_LIBRARY_PATH': '/nix/store/fake/lib'}
            with redirect_stdout(io.StringIO()), patch.object(Path, 'cwd', return_value=root), patch.object(Path, 'is_file', lambda path: True if str(path).startswith('/nix/store/fake/') else original_is_file(path)), patch.object(runner, 'run', side_effect=fake_run), patch.object(runner, 'version', return_value='pinned version'), patch.object(runner.shutil, 'which', return_value='/fake/bin'), patch.object(runner.subprocess, 'check_output', side_effect=lambda cmd, **kw: '' if '--porcelain' in cmd else 'a' * 40), patch.dict(os.environ, env), patch.object(sys, 'argv', ['runner', 'all', '--measure-only', '--artifacts', str(root / 'artifacts')]):
                self.assertEqual(runner.main(), 1)
            manifest = json.loads((root / 'artifacts/manifest.json').read_text())
            self.assertEqual(len(calls), 10)
            self.assertIn('ranges_uncovered', calls)
            self.assertEqual(manifest['scopes']['ranges']['covered_returncode'], 1)
            self.assertTrue((root / 'artifacts/ranges/covered.profraw').is_file())
            self.assertTrue((root / 'artifacts/ranges/inclusive.log').is_file())
            self.assertEqual(manifest['scopes']['dag']['comparison']['status'], 'measured')
            self.assertTrue((root / 'artifacts/summary.md').is_file())


if __name__ == '__main__':
    unittest.main()
