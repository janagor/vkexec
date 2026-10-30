"""Exercise Mull runner discovery without installing Mull or Clang."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == "posix" and shutil.which("cmake"), "requires POSIX and CMake")
class RunnerDiscoveryTests(unittest.TestCase):
    def configure(self, runner_name=None, setting=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plugin = root / "plugin.so"
            plugin.touch()
            profdata = root / "llvm-profdata"
            profdata.write_text('#!/bin/sh\necho "LLVM version 22.1.8"\n')
            profdata.chmod(0o755)
            if runner_name:
                runner = root / runner_name
                runner.write_text("#!/bin/sh\nexit 0\n")
                runner.chmod(0o755)
            module = Path(__file__).resolve().parents[1] / "cmake/MutationTesting.cmake"
            cache = "" if setting is None else f'set(vkexec_MULL_RUNNER "{setting}" CACHE FILEPATH "runner")\n'
            script = root / "test.cmake"
            script.write_text(
                'cmake_minimum_required(VERSION 3.25)\n'
                'set(PROJECT_IS_TOP_LEVEL TRUE)\nset(BUILD_TESTING TRUE)\n'
                'set(CMAKE_CXX_COMPILER_ID Clang)\nset(CMAKE_CXX_COMPILER_VERSION 22.1.8)\n'
                'set(CMAKE_BUILD_TYPE Debug)\n'
                f'set(vkexec_MULL_PLUGIN "{plugin}")\n'
                f'set(vkexec_MULL_PROFDATA "{profdata}")\n'
                # Isolate discovery from installed runners on the host.
                'set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)\n'
                'set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)\n'
                f'set(CMAKE_PROGRAM_PATH "{root}")\n'
                + cache
                + f'include("{module}")\nvkexec_setup_mutation_testing()\n'
                + 'message(STATUS "RUNNER=${vkexec_MULL_RUNNER}")\n'
            )
            result = subprocess.run([shutil.which("cmake"), "-P", str(script)], capture_output=True, text=True)
            return result, str(root / runner_name) if runner_name else None

    def test_unset_and_empty_settings_discover_both_names(self):
        for setting in (None, ""):
            for name in ("mull-runner-22", "mull-runner"):
                with self.subTest(setting=setting, name=name):
                    result, expected = self.configure(name, setting)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"RUNNER={expected}", result.stdout)

    def test_missing_runner_fails(self):
        result, _ = self.configure(setting="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Could not find vkexec_MULL_RUNNER", result.stderr)

    def test_invalid_explicit_path_fails(self):
        for setting in ("/nonexistent/vkexec-mull-runner", "/tmp"):
            with self.subTest(setting=setting):
                result, _ = self.configure("mull-runner-22", setting)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must point to an existing Mull runner", result.stderr)

    def test_explicit_path_is_preserved(self):
        result, _ = self.configure(setting=shutil.which("cmake"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"RUNNER={shutil.which('cmake')}", result.stdout)


if __name__ == "__main__":
    unittest.main()
